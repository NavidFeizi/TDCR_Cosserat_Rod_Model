import torch
import math


class MathOperation:

    def hat(self, v):
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

    def hatSqr(self, v):
        return torch.bmm(self.hat(v), self.hat(v))

    def getSO3(self, h):
        if h.ndim == 3 and h.shape[-1] == 1:
            h = h.squeeze(-1)  # Convert (batch, 3, 1) → (batch, 3)

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

    def quaternionDiff(self, u, h):
        omega = torch.cat([torch.zeros((u.shape[0], 1), device=u.device), u], dim=1)
        return 0.5 * self.quaternion_multiply(h, omega)

    def quaternion_multiply(self, q, r):
        # q, r: shape [B, 4]
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
        )  # shape: [B, 4]
        return result


class TDCR_Physics(MathOperation):
    def __init__(
        self,
        E,
        G,
        radius,
        mass,
        length,
        tendon_offset,
        num_tendons,
        DEVICE=torch.device("cpu"),
        DTYPE=torch.float32,
    ):
        super().__init__()
        self._length = length
        self._tendon_offset = tendon_offset
        self._num_tendons = num_tendons
        self._device = DEVICE
        self._dtype = DTYPE

        gravity = torch.tensor(
            [0.0, -9.8, 0.0], device=self._device, dtype=self._dtype
        )

        # Calculate second moments of area and cross-sectional area
        Ixx = (math.pi / 4) * radius**4
        Izz = 2 * Ixx
        Area = math.pi * radius**2

        # Bending/torsion stiffness matrix
        self.Kbt = torch.diag(
            torch.tensor(
                [E * Ixx, E * Ixx, G * Izz], device=self._device, dtype=self._dtype
            )
        )
        # Shear/extension stiffness matrix
        self.Kse = torch.diag(
            torch.tensor(
                [G * Area, G * Area, E * Area], device=self._device, dtype=self._dtype
            )
        )

        self.norm_kbt = torch.linalg.norm(self.Kbt, ord=2)
        self.norm_kse = torch.linalg.norm(self.Kse, ord=2)

        rho = mass / self._length / Area
        self.m_f = rho * Area * gravity
        self.m_l = torch.zeros(3, device=self._device, dtype=self._dtype)

        # Compute tendon routing positions around the backbone cross-section
        angles = torch.tensor(
            [2 * math.pi * idx / num_tendons for idx in range(num_tendons)],
            device=self._device,
            dtype=self._dtype,
        )
        # m_r: (num_tendons, 3) - position vectors of tendons in local frame
        self.m_r = torch.stack(
            [
                tendon_offset * torch.cos(angles),
                tendon_offset * torch.sin(angles),
                torch.zeros(num_tendons, device=self._device, dtype=self._dtype),
            ],
            dim=1,
        )
        self.m_r = torch.where(
            self.m_r.abs() < 1e-5, torch.zeros_like(self.m_r), self.m_r
        )

        # Reference strains for the backbone (straight, no twist)
        self.v_star = torch.tensor(
            [0.0, 0.0, 1.0], device=self._device, dtype=self._dtype
        ).unsqueeze(1)
        self.u_star = torch.tensor(
            [0.0, 0.0, 0.0], device=self._device, dtype=self._dtype
        ).unsqueeze(1)

    def estimate_initial_value(self, tau):
        """This member function is not batched"""
        # tau shape: (num_tendon)

        p0 = torch.tensor([0.0, 0.0, 0.0], device=self._device, dtype=tau.dtype)
        h0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device, dtype=tau.dtype)
        R0 = self.getSO3(h0.unsqueeze(0)).squeeze(0)

        n0 = torch.zeros((3, 1), device=self._device, dtype=tau.dtype)
        m0 = torch.zeros((3, 1), device=self._device, dtype=tau.dtype)

        for idx in range(self._num_tendons):
            fb_i = -tau[idx].unsqueeze(0) * (self.v_star / torch.norm(self.v_star))
            n0 += fb_i
            m0 += torch.cross(self.m_r[idx].unsqueeze(1), fb_i, dim=0)

        v0 = torch.inverse(self.Kse) @ (torch.inverse(R0) @ n0) + self.v_star
        u0 = torch.inverse(self.Kbt) @ (torch.inverse(R0) @ m0) + self.u_star

        return torch.cat([p0, h0, v0.squeeze(1), u0.squeeze(1)], dim=0)

    def ode_function(self, y, tau):
        batch_size = y.shape[0]
        p = y[:, 0:3].unsqueeze(2)
        h = y[:, 3:7]
        v = y[:, 7:10].unsqueeze(2)
        u = y[:, 10:13].unsqueeze(2)

        R = self.getSO3(h)
        uh = self.hat(u)

        A, B, G, H = [
            torch.zeros((batch_size, 3, 3), device=y.device, dtype=y.dtype)
            for _ in range(4)
        ]
        a, b = [
            torch.zeros((batch_size, 3, 1), device=y.device, dtype=y.dtype)
            for _ in range(2)
        ]

        for idx in range(self._num_tendons):
            m_r_batch = self.m_r[idx].repeat(batch_size, 1).unsqueeze(2)
            pbi_s = torch.bmm(uh, m_r_batch) + v
            pbi_sNorm = torch.norm(pbi_s, dim=1)
            Ai = -(tau[:, idx].unsqueeze(1) / (pbi_sNorm**3)).view(
                -1, 1, 1
            ) * self.hatSqr(pbi_s)
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
        dhds = self.quaternionDiff(u.squeeze(2), h)

        dyds = torch.cat(
            [dpds.squeeze(2), dhds, lhs.squeeze(2)], dim=1
        )
        return dyds

    def boundary_function(self, ya, yb, tau):
        batch_size = yb.shape[0]
        # ya shape: (Batch, 13) # ya: state at s = 0
        # yb shape: (Batch, 13) # yb: state at s = L
        # tau shape: (Batch, 4)

        # Extract start position and orientation quaternion at base
        p0_error = ya[:, 0:3] - torch.tensor(
            [0.0, 0.0, 0.0], device=self._device, dtype=yb.dtype
        ).unsqueeze(0).expand(batch_size, 3)
        h0_error = ya[:, 3:7] - torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self._device, dtype=yb.dtype
        ).unsqueeze(0).expand(batch_size, 4)

        # tau_batch = tau.unsqueeze(0).expand(batch_size, self.num_tendons)

        # Compute tip force/moment in body frame
        vl, ul = yb[:, 7:10].unsqueeze(2), yb[:, 10:13].unsqueeze(2)

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
        for idx in range(self._num_tendons):
            m_r_batch = self.m_r[idx].repeat(batch_size, 1).unsqueeze(2)
            u1_hat = self.hat(ul)
            pbi_s = torch.bmm(u1_hat, m_r_batch) + vl
            pbi_s_norm = torch.norm(pbi_s, dim=1)
            fb_i = -tau[:, idx : idx + 1].unsqueeze(2) * (
                pbi_s / pbi_s_norm.unsqueeze(-1)
            )
            force_err = force_err + fb_i
            moment_err = moment_err + torch.cross(m_r_batch, fb_i, dim=1)

        # residual = torch.cat([p0_error, h0_error, nb.squeeze(2)*1e-3], dim=1)
        residual = torch.cat(
            [p0_error, h0_error, force_err.squeeze(2), moment_err.squeeze(2)], dim=1
        )
        return residual


