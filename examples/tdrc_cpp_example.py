import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), "/build/lib/libtdcr.so"))

import tdcr_cpp


tdcr = tdcr_cpp.TDCR(
    15.467e9,  # E
    5.6e9,     # G
    0.0004,    # radius
    0.00516,   # mass
    0.05436,   # length
    1.112e-3   # tendon offset
)


tdcr.update_point_force([0.0, 0.0, 0.0])
tdcr.update_initial_guess([0.0, 10.0, 0.0, 0.0])
tdcr.set_tendon_pull([0.0, 10.0, 0.0, 0.0])
tdcr.solve_bvp()

backbone = tdcr.get_backbone()
base_state = tdcr.get_base_state()
tip_pos = tdcr.get_tip_pos()


# Example usage:
# tdcr_wrapper = TDCRWrapper(tdcr)
# result = tdcr_wrapper([0.0, 10.0, 0.0, 0.0])
# print("Result from wrapper:", result)

print("Backbone first point:", backbone[0])
print("Tip position:", tip_pos)