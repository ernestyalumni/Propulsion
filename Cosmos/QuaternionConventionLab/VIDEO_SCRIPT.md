# Quaternion Convention Lab — video package

## Recommended title and packaging

**YouTube title:** Why Quaternion Conventions Break Spacecraft Simulations

**Subtitle / description hook:** `q` and `−q` are the same physical attitude—but `[w,x,y,z]` and `[x,y,z,w]` are not interchangeable interfaces.

**Thumbnail:** two visually aligned spacecraft labeled `q` and `−q`, one divergent spacecraft in red, with **SAME ROTATION. WRONG SPACECRAFT.**

Avoid putting “Hamilton vs JPL” in the title. “JPL convention” is used inconsistently; the video’s point is to specify the entire contract.

## Five-minute YouTube script

### 0:00–0:20 — cold open

*[Convention mismatch mode. Two spacecraft visibly disagree.]*

“Both of these quaternions are normalized. Both are made of the same four numbers. And both code paths compile. But one spacecraft is pointing in the wrong direction. This is the kind of bug that can survive unit tests—unless your tests state what a quaternion actually means.”

### 0:20–0:55 — the two different “twos”

*[Show slide 1.]*

“People often combine two separate ideas. First, unit quaternions double-cover 3D rotations: `q` and `−q` are different points on the unit 3-sphere but the same element of SO(3). Second, engineering libraries use different conventions: scalar first or last, active or passive, frame direction, multiplication definition, and composition order. The double cover causes the first ambiguity. It does not cause the second.”

### 0:55–1:45 — show the double cover

*[Switch to q versus −q and play 0° through 720°.]*

“Here the left model uses `q`; the right uses `−q`. Their components disagree in sign, but their physical attitude error is zero because `R(q)` equals `R(−q)`. Follow a continuous rotation: after 360 degrees, the quaternion reaches the antipode, approximately `−1,0,0,0`, although the spacecraft is physically back at identity. At 720 degrees, it returns to the original quaternion representative.”

“The software consequence is practical. Never use raw component distance as an attitude metric. Use a sign-invariant metric such as `2 acos(abs(dot(q1,q2)))`. Before interpolation or differencing a time history, align adjacent samples to the same hemisphere.”

### 1:45–2:50 — inject convention failures

*[Select scalar-layout mismatch.]*

“Now I transmit scalar-last bytes `[x,y,z,w]` and deliberately consume them through a scalar-first constructor `[w,x,y,z]`. Nothing is unnormalized. The error can still be enormous. The correct fix is a named adapter at the interface—not a mysterious sign flip in the dynamics.”

*[Select active/passive mismatch.]*

“Next, one side treats the quaternion as an active body-to-world rotation and the other as the inverse passive coordinate transform. In this contract, that conjugates the quaternion and reverses the rotation. Saying only ‘we use Hamilton quaternions’ would not resolve this.”

### 2:50–3:50 — the executable contract

*[Show slide 2, then C++ and Rust side-by-side.]*

“My contract is explicit: Hamilton product, scalar-first API, active body-to-world mapping, and the sandwich product `q (0,v) q-star`. C++20 and Rust implement the same semantics without external math dependencies.”

“The most valuable test is almost embarrassingly small: positive 90 degrees about positive Z must map positive X to positive Y. That one golden vector catches action direction, handedness, and sign. I also assert that `q` and `−q` produce identical rotation matrices, round-trip layout adapters, and inject failures to prove the tests can fail.”

### 3:50–4:35 — connect it to a spacecraft simulator

“In a six-degree-of-freedom simulation, the contract sits exactly at the boundary between body-frame torques and inertial-frame translation. The attitude propagator needs a declared angular-velocity frame and multiplication side. Gravity-gradient torque starts with the local vertical expressed in body coordinates. Aerodynamic drag uses atmosphere-relative velocity in the inertial translational dynamics. A quaternion ambiguity at either boundary gives plausible-looking but physically inconsistent coupled behavior.”

### 4:35–5:00 — close

*[Return to aligned q/−q models, then flip to mismatch.]*

“So the concise answer is: `q` and `−q` come from SU(2) double-covering SO(3). Quaternion conventions are separate interface choices. Name all of them, adapt at boundaries, and test a known physical rotation. A quaternion is only four numbers after its semantics are fixed.”

## 45–60 second short

*[Open on aligned spacecraft.]*

“These spacecraft use `q` and `−q`. Different quaternion, same physical rotation. Why? Unit quaternions form SU(2), which double-covers SO(3), so `R(q) = R(−q)`.”

*[Switch to convention mismatch; spacecraft diverge.]*

“But this is a different problem. One system writes `[x,y,z,w]`; another reads `[w,x,y,z]`. Or one means active body-to-world while another means passive world-to-body. Those aren’t caused by the double cover. They’re software interface contracts—and normalized inputs can still be completely wrong.”

*[Flash code test.]*

“State the algebra, layout, action, frame direction, and composition order. Then pin them down with one test: positive 90 degrees about Z maps X to Y. `q` and `−q`: same attitude. Undeclared convention: wrong spacecraft.”

## Recording checklist

- Record 16:9 at 1440p; crop a second 9:16 pass around the spacecraft and error readout.
- Capture the browser demo first; add slide/code inserts in editing.
- Keep the live demo under 90 seconds in any interview setting.
- Do not call scalar-last universally “the JPL convention.” Say exactly which layout and action the boundary expects.
- Put the repository path and explicit convention contract in the description.
