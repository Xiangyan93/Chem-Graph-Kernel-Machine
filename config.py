import os
from graphdot.microkernel import (
    Additive,
    Normalize,
    Constant as kC,
    TensorProduct,
    SquareExponential,
    KroneckerDelta as kDelta,
    Convolution as kConv,
)


class Config:
    CWD = os.path.dirname(os.path.abspath(__file__))

    # tensorproduct
    class Hyperpara:  # initial hyperparameter used in graph kernel
        k = 0.90
        q = 0.01  # q is the stop probability in ramdom walk
        k_bounds = (0.1, 1.0)
        s_bounds = (0.1, 10.0)
        q_bound = (1e-4, 1.0)
        knode = TensorProduct(
            atomic_number=kDelta(0.75, k_bounds),
            aromatic=kDelta(k, k_bounds),
            charge=SquareExponential(
                length_scale=2.5,
                length_scale_bounds=s_bounds
            ),
            hcount=SquareExponential(
                length_scale=2.5,
                length_scale_bounds=s_bounds
            ),
            chiral=kDelta(k, k_bounds),
            ring_list=kConv(kDelta(k, k_bounds)),
            morgan_hash=kDelta(k, k_bounds),
            ring_number=kDelta(k, k_bounds),
            # hybridization=kDelta(k, k_bounds),
        )
        kedge = TensorProduct(
            order=SquareExponential(
                length_scale=1.5,
                length_scale_bounds=s_bounds
            ),
            # aromatic=kDelta(k, k_bounds),
            stereo=kDelta(k, k_bounds),
            conjugated=kDelta(k, k_bounds),
            ring_stereo=kDelta(k, k_bounds),
            # symmetry=kDelta(k, k_bounds),
        )

    # additive
    class Hyperpara1:
        k = 0.90
        q = 0.01  # q is the stop probability in ramdom walk
        k_bounds = (0.1, 1.0)
        s_bounds = (0.1, 10.0)
        q_bound = (1e-4, 1.0)
        knode = Normalize(
            Additive(
                atomic_number=kC(0.5, (0.1, 1.0)) * kDelta(0.75, k_bounds),
                aromatic=kC(0.5, (0.1, 1.0)) * kDelta(k, k_bounds),
                charge=kC(0.5, (0.1, 1.0)) * SquareExponential(
                    length_scale=2.5,
                    length_scale_bounds=s_bounds
                ),
                hcount=kC(0.5, (0.1, 1.0)) * SquareExponential(
                    length_scale=2.5,
                    length_scale_bounds=s_bounds
                ),
                chiral=kC(0.5, (0.1, 1.0)) * kDelta(k, k_bounds),
                ring_list=kC(0.5, (0.1, 1.0)) * kConv(kDelta(k, k_bounds)),
                morgan_hash=kC(0.5, (0.1, 1.0)) * kDelta(k, k_bounds),
                ring_number=kC(0.5, (0.1, 1.0)) * kDelta(k, k_bounds),
            )
        )
        kedge = Normalize(
            Additive(
                order=kC(0.5, (0.1, 1.0)) * SquareExponential(
                    length_scale=1.5,
                    length_scale_bounds=s_bounds
                ),
                # aromatic=kDelta(k, k_bounds),
                stereo=kC(0.5, (0.1, 1.0)) * kDelta(k, k_bounds),
                conjugated=kC(0.5, (0.1, 1.0)) * kDelta(k, k_bounds),
                # ring_stereo=kC(0.5, (0.1, 1.0)) * kDelta(k, k_bounds),
                # symmetry=kDelta(k, k_bounds),
            )
        )


