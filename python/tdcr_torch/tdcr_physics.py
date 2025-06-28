import torch
import math


class MathOperation:
    """
    A utility class providing mathematical operations for vector and quaternion algebra,
    - All operations are batched.
    - Input tensors are of appropriate shapes as described in each method.
    - For use with PyTorch tensors.
    """

    def hat(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the skew-symmetric matrix (hat operator) for a batch of 3D vectors.
        Args:
            v (torch.Tensor): Shape (batch, 3) or (batch, 3, 1).
        Returns:
            torch.Tensor: Shape (batch, 3, 3)
        """
        if v.ndim == 3 and v.shape[-1] == 1:
            v = v.squeeze(-1)  # Convert (batch, 3, 1) → (batch, 3)

        zero = torch.zeros((v.shape[0]), device=v.device, dtype=v.dtype)
        hat = torch.stack(
            [
                torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
                torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
                torch.stack([-v[:, 1], v[:, 0], zero], dim=1),
            ],
            dim=1,
        )
        return hat

    def hat_sqr(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the square of the skew-symmetric (hat) matrix of 3D vectors.
        Args:
            v (torch.Tensor): Shape (batch, 3) or (batch, 3, 1).
        Returns:
            torch.Tensor: Shape (batch, 3, 3)
        """
        return torch.bmm(self.hat(v), self.hat(v))

    def get_SO3(self, h: torch.Tensor) -> torch.Tensor:
        """
        Converts quaternions to their corresponding SO(3) rotation matrices.
        Args:
            h (torch.Tensor): Shape (batch, 4) or (batch, 4, 1), quaternions in the format [qw, qx, qy, qz].
        Returns:
            torch.Tensor: Shape (batch, 3, 3), Rotation matrices.
        """
        if h.ndim == 3 and h.shape[-1] == 1:
            h = h.squeeze(-1)  # Convert (batch, 4, 1) → (batch, 4)

        # q shape: (Batch, 4)
        q0, q1, q2, q3 = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
        R = torch.stack(
            [
                torch.stack(
                    [
                        1 - 2 * (q2**2 + q3**2),
                        2 * (q1 * q2 - q0 * q3),
                        2 * (q1 * q3 + q0 * q2),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        2 * (q1 * q2 + q0 * q3),
                        1 - 2 * (q1**2 + q3**2),
                        2 * (q2 * q3 - q0 * q1),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        2 * (q1 * q3 - q0 * q2),
                        2 * (q2 * q3 + q0 * q1),
                        1 - 2 * (q1**2 + q2**2),
                    ],
                    dim=1,
                ),
            ],
            dim=1,
        )  # shape (B, 3, 3)
        return R

    def quaternion_diff(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the time derivative of a quaternion given the angular velocity.
        Args:
            u (torch.Tensor): Shape (batch, 3), Angular velocity vector.
            h (torch.Tensor): Shape (batch, 4), Orientation quaternion in the format [qw, qx, qy, qz].
        Returns:
            torch.Tensor: Shape (batch, 4).
        """
        omega = torch.cat([torch.zeros((u.shape[0], 1), device=u.device), u], dim=1)
        return 0.5 * self.quaternion_multiply(h, omega)

    def quaternion_multiply(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Multiplies two batches of quaternions.
        Given two tensors `q` and `r` of shape (batch, 4), representing batches of quaternions in the format[w, x, y, z],
        computes the Hamilton product for each pair in the batch.
        Args:
            q (torch.Tensor): Shape (batch, 4), quaternions.
            r (torch.Tensor): Shape (batch, 4), quaternions.
        Returns:
            torch.Tensor: Shape (batch, 4).
        """
        # q, r: shape [batch, 4]
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        r0, r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        result = torch.stack(
            [
                q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3,
                q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2,
                q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1,
                q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0,
            ],
            dim=1,
        )  # shape: [batch, 4]
        return result


class TDCR_Physics(MathOperation):
    """
    The physics model for the statics of Tendon-Driven Continuum Robot (TDCR) using Cosserat rod theory implemented in PyTorch with automatic differentiation with batch enabled.

    based on the papers:
    - Rucker, D. Caleb, and Robert J. Webster III. "Statics and dynamics of continuum robots with general tendon routing and external loading." IEEE Transactions on Robotics 27.6 (2011): 1033-1044.
    - Till, John, Vincent Aloi, and Caleb Rucker. "Real-time dynamics of soft and continuum robots based on Cosserat rod models." The International Journal of Robotics Research 38.6 (2019): 723-746.

    Parameters
    ----------
    E : float
        Young's modulus of the rod material.
    G : float
        Shear modulus of the rod material.
    radius : float
        Radius of the rod's cross-section.
    mass : float
        Total mass of the rod.
    length : float
        Length of the rod.
    tendon_offset : float
        Radial distance from the rod center to the tendon routing.
    num_tendons : int
        Number of tendons actuating the rod.
    device : torch.device, optional
        PyTorch device for tensor computations (default: torch.device("cpu")).
    dtype : torch.dtype, optional
        PyTorch data type for tensors (default: torch.float32).
    """

    def __init__(
        self,
        E: float,
        G: float,
        radius: float,
        mass: float,
        length: float,
        tendon_offset: float,
        num_tendons: int,
        device=torch.device("cpu"),
        dtype=torch.float64,
    ):
        super().__init__()
        self.length = length
        self.tendon_offset = tendon_offset
        self.num_tendons = num_tendons
        self.device = device
        self.dtype = dtype

        gravity = torch.tensor([0.0, -9.8, 0.0], device=self.device, dtype=self.dtype)

        # Calculate second moments of area and cross-sectional area
        Ixx = (math.pi / 4) * radius**4
        Izz = 2 * Ixx
        Area = math.pi * radius**2

        # Bending/torsion stiffness matrix
        self.Kbt = torch.diag(
            torch.tensor(
                [E * Ixx, E * Ixx, G * Izz], device=self.device, dtype=self.dtype
            )
        )
        # Shear/extension stiffness matrix
        self.Kse = torch.diag(
            torch.tensor(
                [G * Area, G * Area, E * Area], device=self.device, dtype=self.dtype
            )
        )

        # norm of stiffness matrices used for scaling for deeplearning applications
        self.norm_kbt = torch.linalg.norm(self.Kbt, ord=2)
        self.norm_kse = torch.linalg.norm(self.Kse, ord=2)

        # mass density
        rho = mass / self.length / Area
        self.m_f = rho * Area * gravity
        self.m_l = torch.zeros(3, device=self.device, dtype=self.dtype)

        # Compute tendon routing positions around the backbone cross-section
        angles = torch.tensor(
            [2 * math.pi * idx / num_tendons for idx in range(num_tendons)],
            device=self.device,
            dtype=self.dtype,
        )
        # m_r: (num_tendons, 3) - position vectors of tendons in local frame
        self.m_r = torch.stack(
            [
                tendon_offset * torch.cos(angles),
                tendon_offset * torch.sin(angles),
                torch.zeros(num_tendons, device=self.device, dtype=self.dtype),
            ],
            dim=1,
        )
        # Ensure m_r is not too small to avoid numerical issues
        self.m_r = torch.where(
            self.m_r.abs() < 1e-5, torch.zeros_like(self.m_r), self.m_r
        )

        # Reference strains for the backbone (straight, no twist)
        self.v_star = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype
        ).unsqueeze(1)
        self.u_star = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device, dtype=self.dtype
        ).unsqueeze(1)

    def ode_function(self, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Computes the derivatives state for use in ODE integration.

        Args:
            y (torch.Tensor): The current state tensor of shape (batch, 13), where each row contains:
                - p (3): Position vector.
                - h (4): Orientation quaternion.
                - v (3): Linear strain vector.
                - u (3): Angular strain vector.
            tau (torch.Tensor): Tendon tension tensor of shape (batch, num_tendons).

        Returns:
            torch.Tensor: Shape (batch_size, 13).
        """

        batch_size = y.shape[0]
        p = y[:, 0:3].unsqueeze(2)
        h = y[:, 3:7]
        v = y[:, 7:10].unsqueeze(2)
        u = y[:, 10:13].unsqueeze(2)

        R = self.get_SO3(h)
        uh = self.hat(u)

        A, B, G, H = [
            torch.zeros((batch_size, 3, 3), device=y.device, dtype=y.dtype)
            for _ in range(4)
        ]
        a, b = [
            torch.zeros((batch_size, 3, 1), device=y.device, dtype=y.dtype)
            for _ in range(2)
        ]

        for idx in range(self.num_tendons):
            m_r_batch = self.m_r[idx].repeat(batch_size, 1).unsqueeze(2)
            pbi_s = torch.bmm(uh, m_r_batch) + v
            pbi_sNorm = torch.norm(pbi_s, dim=1)
            Ai = -(tau[:, idx].unsqueeze(1) / (pbi_sNorm**3)).view(
                -1, 1, 1
            ) * self.hat_sqr(pbi_s)
            hat_m_r = self.hat(m_r_batch)
            Bi = torch.bmm(hat_m_r, Ai)
            A = A + Ai
            B = B + Bi
            G = G - torch.bmm(Ai, hat_m_r)
            H = H - torch.bmm(Bi, hat_m_r)
            ai = torch.bmm(Ai, torch.bmm(uh, pbi_s))
            bi = torch.bmm(hat_m_r, ai)
            a = a + ai
            b = b + bi

        v_star_exp = self.v_star.repeat(batch_size, 1, 1)
        u_star_exp = self.u_star.repeat(batch_size, 1, 1)
        Kse_exp = self.Kse.unsqueeze(0).expand(batch_size, -1, -1)
        Kbt_exp = self.Kbt.unsqueeze(0).expand(batch_size, -1, -1)
        nb = torch.bmm(Kse_exp, (v - v_star_exp))
        mb = torch.bmm(Kbt_exp, (u - u_star_exp))

        m_f_exp = self.m_f.repeat(batch_size, 1).unsqueeze(2)
        m_l_exp = self.m_l.repeat(batch_size, 1).unsqueeze(2)

        rhs_upper = -torch.bmm(uh, nb) - torch.bmm(R.transpose(-1, -2), m_f_exp) - a
        rhs_lower = (
            -torch.bmm(uh, mb)
            - torch.bmm(self.hat(v), nb)
            - torch.bmm(R.transpose(-1, -2), m_l_exp)
            - b
        )
        rhs = torch.cat([rhs_upper, rhs_lower], dim=1)

        phi_upper = torch.cat([Kse_exp + A, G], dim=2)
        phi_lower = torch.cat([B, Kbt_exp + H], dim=2)
        phi = torch.cat([phi_upper, phi_lower], dim=1)

        lhs = torch.linalg.solve(phi, rhs)

        dpds = torch.bmm(R, v)
        dhds = self.quaternion_diff(u.squeeze(2), h)

        dyds = torch.cat([dpds.squeeze(2), dhds, lhs.squeeze(2)], dim=1)
        return dyds

    def boundary_function(
        self, ya: torch.Tensor, yb: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the boundary residuals for the BVP.

        Parameters
        ----------
        ya : torch.Tensor
            State tensor at the base (s = 0) of shape (batch, 13)
        yb : torch.Tensor
            State tensor at the tip (s = L) of shape (batch, 13).
        tau : torch.Tensor
            Tendon tensions of shape (batch, num_tendons).

        Returns
        -------
        torch.Tensor
            Residual tensor of shape (batch, 13), concatenating:
            - Position error at base (3,)
            - Orientation quaternion error at base (4,)
            - Force error at tip (3,)
            - Moment error at tip (3,)
        """
        batch_size = yb.shape[0]

        # Extract start position and orientation quaternion at base
        p0_error = ya[:, 0:3] - torch.tensor(
            [0.0, 0.0, 0.0], device=self.device, dtype=yb.dtype
        ).unsqueeze(0).expand(batch_size, 3)
        h0_error = ya[:, 3:7] - torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=yb.dtype
        ).unsqueeze(0).expand(batch_size, 4)

        vl = yb[:, 7:10].unsqueeze(2)
        ul = yb[:, 10:13].unsqueeze(2)

        # Compute tip force/moment in body frame
        nb = torch.bmm(
            self.Kse.unsqueeze(0).expand(batch_size, 3, 3),
            (vl - self.v_star.unsqueeze(0).expand(batch_size, 3, 1)),
        )
        mb = torch.bmm(
            self.Kbt.unsqueeze(0).expand(batch_size, 3, 3),
            (ul - self.u_star.unsqueeze(0).expand(batch_size, 3, 1)),
        )
        force_err = -nb
        moment_err = -mb

        # Add tendon contributions as in bc_function
        for idx in range(self.num_tendons):
            m_r_batch = self.m_r[idx].repeat(batch_size, 1).unsqueeze(2)
            u1_hat = self.hat(ul)
            pbi_s = torch.bmm(u1_hat, m_r_batch) + vl
            pbi_s_norm = torch.norm(pbi_s, dim=1)
            fb_i = -tau[:, idx : idx + 1].unsqueeze(2) * (
                pbi_s / pbi_s_norm.unsqueeze(-1)
            )
            force_err = force_err + fb_i
            moment_err = moment_err + torch.cross(m_r_batch, fb_i, dim=1)

        residual = torch.cat(
            [p0_error, h0_error, force_err.squeeze(2), moment_err.squeeze(2)], dim=1
        )
        return residual

    def estimate_initial_value(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Estimate the initial state vector for the Cosserat rod model given tendon tensions.
        This function is not batched.
        Args:
            tau (torch.Tensor): Shape (num_tendon,)

        Returns:
            torch.Tensor: Shape (13,) initial state vector:
                [position (3,), orientation quaternion (4,), internal force (3,), internal moment (3,)].
        """

        p0 = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=tau.dtype)
        h0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=tau.dtype)
        R0 = self.get_SO3(h0.unsqueeze(0)).squeeze(0)

        n0 = torch.zeros((3, 1), device=self.device, dtype=tau.dtype)
        m0 = torch.zeros((3, 1), device=self.device, dtype=tau.dtype)

        for idx in range(self.num_tendons):
            fb_i = -tau[idx].unsqueeze(0) * (self.v_star / torch.norm(self.v_star))
            n0 += fb_i
            m0 += torch.cross(self.m_r[idx].unsqueeze(1), fb_i, dim=0)

        v0 = torch.inverse(self.Kse) @ (torch.inverse(R0) @ n0) + self.v_star
        u0 = torch.inverse(self.Kbt) @ (torch.inverse(R0) @ m0) + self.u_star

        return torch.cat([p0, h0, v0.squeeze(1), u0.squeeze(1)], dim=0)

    def reduced_ode_function(self, y_reduced: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        batch_size = y_reduced.shape[0]
        y = torch.zeros(batch_size, 13, device=y_reduced.device, dtype=y_reduced.dtype)
        y[:, 0:3] = y_reduced[:, 0:3]
        y[:, 3:7] = y_reduced[:, 3:7]
        y[:, 7:10] = torch.tensor([0.0, 0.0, 1.0], device=y_reduced.device, dtype=y_reduced.dtype).unsqueeze(0).expand(batch_size, 3)
        y[:, 10:13] = y_reduced[:, 7:10]
        return self.ode_function(y.unsqueeze(0), tau).squeeze(0)
