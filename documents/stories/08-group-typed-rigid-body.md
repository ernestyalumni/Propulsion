As an engineer composing simulations across domains, I want every rigid-body quantity in the library to declare the group it lives in and the conventions it assumes, so that two modules cannot silently disagree about what a rotation means.

Every type representing an attitude MUST declare that it lives in SO(3). Every type representing a full six-degree-of-freedom pose MUST declare that it lives in SE(3). Every quaternion type MUST declare that unit quaternions form SU(2) and double-cover SO(3).

Every quaternion API boundary MUST name all five conventions fixed in Cosmos/QuaternionConventionLab/README.md: multiplication, scalar layout, active or passive action, frame direction, and composition order.

The library MUST hold exactly one quaternion convention. A module MUST NOT introduce a second one; a module that must interoperate with another convention MUST convert through a named adapter rather than redefining the boundary.

Conversions between representations MUST be property-tested against the double cover: a quaternion and its negation MUST produce the same rotation matrix, and a round trip through rotation matrix and back MUST return the original quaternion up to sign.

These declarations MUST cite LaTeXandpdfs/SO3_SU2_Quaternions.tex for the derivation, and MUST NOT restate it.

Never widen a declared group to make a conversion typecheck.

For example: Cosmos/QuaternionConventionLab already fixes Hamilton multiplication, scalar-first layout, active rotation, body-to-world frame direction and q2*q1 composition order, and carries dependency-free C++ and Rust kernels that hold to it. That same contract MUST hold in Cosmos/Source and in CombustionInstability, and a test MUST fail if a module there uses scalar-last layout.
