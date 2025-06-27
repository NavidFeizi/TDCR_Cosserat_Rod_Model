import numpy as np
import torch
from scipy.integrate import solve_bvp
from .tdcr_physics import TDCR_Physics


## TDCR with torch backend


device = torch.device("cpu")
torch.set_num_threads(12)


class TDCR_Robot(TDCR_Physics):
    def __init__(self, E, G, radius, mass, length, tendon_offset, num_tendons, DEVICE, s_steps = 200):

        super().__init__(
            E=E,
            G=G,
            radius=radius,
            mass=mass,
            length=length,
            tendon_offset=tendon_offset,
            num_tendons=num_tendons,
            DEVICE=DEVICE,
            DTYPE=torch.float64,
        )
        self.length = length
        self.num_tendons = num_tendons
        self.s_steps = s_steps
        self.tau = torch.zeros(num_tendons, dtype=self._dtype, device=self._device)

    def set_tendon_pull(self, tau):
        self.tau = torch.tensor(tau, dtype=self._dtype, device=self._device)
        self.guess = self.estimate_initial_value(self.tau)      

    def bvp_solve(self):
        tol = 1e-5
        max_iter = 1000
        eps = 1e-9

        with torch.no_grad():
            # init_guess = torch.tensor(
            #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=self._dtype, device=self._device
            # )
            init_guess = self.guess[7:]

            for i in range(max_iter):
                residual = self.residual_func(init_guess.unsqueeze(0)).squeeze(0)
                err = residual.norm().item()
                # print(f"Iter {i}, residual norm: {err:.3e}")
                if err < tol:
                    # print("Converged!")
                    break
                jac = self.compute_jacobian(init_guess, eps)
                # delta = torch.linalg.solve(jac, residual)
                delta = torch.inverse(jac) @ residual
                init_guess -= delta

            # print("init_guess: " + str(init_guess))
            # Once converged, integrate final trajectory
            y0 = torch.zeros(13, dtype=self._dtype, device=self._device)
            y0[0:3] = torch.tensor(
                [0.0, 0.0, 0.0], dtype=self._dtype, device=self._device
            )
            y0[3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self._dtype, device=self._device
            )
            y0[7:13] = init_guess
            Y = self.rk4_integrate(y0.unsqueeze(0), self.tau.unsqueeze(0)).squeeze(0)
        return Y.cpu().numpy()

    def compute_jacobian(self, guess, eps=1e-8):
        dim = len(guess)
        jac = torch.zeros((6, 6), dtype=guess.dtype, device=self._device)

        perturbedGuess = guess.expand(dim * 2, dim).clone()
        perturbedGuess[:dim, :] += torch.eye(dim) * eps
        perturbedGuess[dim:, :] -= torch.eye(dim) * eps
        residual_all = self.residual_func(perturbedGuess)
        residual_plus = residual_all[:dim, :].T
        residual_minus = residual_all[dim:, :].T

        jac = 0.5 * (residual_plus - residual_minus) / eps
        return jac
    
    def compute_jacobian_forloop(self, guess, eps=1e-8):
        jac = torch.zeros((6, 6), dtype=guess.dtype, device=self._device)

        residual_plus = torch.zeros((6, 6), dtype=self._dtype, device=self._device)
        residual_minus = torch.zeros((6, 6), dtype=self._dtype, device=self._device)
        for i in range(6):
            perturbedGuess = guess.clone()
            perturbedGuess[i] += eps
            residual = self.residual_func(perturbedGuess.unsqueeze(0))
            residual_plus[:, i] = residual
        # print("residual_plus: " + str(residual_plus))
        for i in range(6):
            perturbedGuess = guess.clone()
            perturbedGuess[i] -= eps
            residual = self.residual_func(perturbedGuess.unsqueeze(0))
            residual_minus[:, i] = residual
        # print("residual_minus: " + str(residual_minus))

        jac = 0.5 * (residual_plus - residual_minus) / eps
        return jac

    def residual_func(self, init_guess):
        # guess: torch tensor (batch, 6) [n0, m0] -- initial force/moment at base
        # Returns: torch tensor (6,)

        batch_size = init_guess.shape[0]
        ### assemble initial state
        y0 = torch.zeros((batch_size, 13), dtype=init_guess.dtype, device=self._device)
        y0[:, 0:3] = (
            torch.tensor([0.0, 0.0, 0.0], dtype=init_guess.dtype, device=self._device)
            .unsqueeze(0)
            .expand(batch_size, 3)
        )
        y0[:, 3:7] = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=init_guess.dtype, device=self._device)
            .unsqueeze(0)
            .expand(batch_size, 4)
        )
        y0[:, 7:13] = init_guess

        tau_batch = self.tau.unsqueeze(0).expand(batch_size, self.num_tendons)

        # Integrate ODE
        Y = self.rk4_integrate(y0, tau_batch)
        y0 = Y[:, :, 0]
        yl = Y[:, :, -1]

        residual = self.boundary_function(ya=y0, yb=yl, tau=self.tau.unsqueeze(0))
        return residual[:, 7:]

    def rk4_integrate(self, y0, tau):
        # y0: (batch, 13), tau: (batch, num_tendons)
        # Returns y: (batch, s_steps, 13)

        ds = self.length / (self.s_steps - 1)
        half_ds = ds / 2.00
        sixth_ds = ds / 6.00
        batch_size = y0.shape[0]
        Y = torch.zeros(
            (batch_size, y0.shape[1], self.s_steps),
            dtype=y0.dtype,
            device=self._device,
        )
        Y[:, :, 0] = y0
        y = y0
        for i in range(0, self.s_steps - 1):
            k1 = self.ode_function(y, tau)
            k2 = self.ode_function(y + half_ds * k1, tau)
            k3 = self.ode_function(y + half_ds * k2, tau)
            k4 = self.ode_function(y + ds * k3, tau)
            y += (sixth_ds * (k1 + 2 * k2 + 2 * k3 + k4)).squeeze(0)
            Y[:, :, i + 1] = y
        return Y  # Remove batch

    def ode_function_numpy(self, s_np, y_np, tau):
        y = torch.tensor(y_np.T, dtype=self._dtype, device=self._device)
        tau = tau.unsqueeze(0).expand(y.shape[0], -1)  # shape: (n_points, num_tendons)
        dyds = super().ode_function(y, tau).numpy() 
        return dyds.T

    def boundary_function_numpy(self, ya_np, yb_np, tau):
        ya = torch.tensor(ya_np, dtype=self._dtype, device=self._device).unsqueeze(0)
        yb = torch.tensor(yb_np, dtype=self._dtype, device=self._device).unsqueeze(0)

        residual = super().boundary_function(ya, yb, tau).squeeze(0).numpy()
        return residual

    def scipy_solver(self):
        s = np.linspace(0.0, self.length, self.s_steps)

        # p_init = np.array([0.0, 0.0, 0.0])
        # h_init = np.array([1.0, 0.0, 0.0, 0.0])
        # v_init = np.array([0.0, 0.0, 1.0])
        # u_init = np.array([0.0, 0.0, 0.0])
        # y0 = np.concatenate((p_init, h_init, v_init, u_init))
        # init_guess = np.tile(y0[:, np.newaxis], (1, self.s_steps))
        init_guess = np.tile(self.guess.numpy()[:, np.newaxis], (1, self.s_steps))

        sol = solve_bvp(
            lambda s, y: self.ode_function_numpy(s, y, self.tau),
            lambda ya, yb: self.boundary_function_numpy(ya, yb, self.tau.unsqueeze(0)),
            s,
            init_guess,
            tol=1e-6,
            max_nodes=10000,
        )
        # s_uniform = np.linspace(0, L, 200)  # fixed-length output
        y_uniform = sol.sol(s)   # interpolated

        # print("Solved")
        return y_uniform
        # return sol.y[:]

