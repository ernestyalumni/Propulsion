(() => {
  "use strict";

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x02070d, 0.025);

  const camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 0.1, 500);
  camera.position.set(0, 3.5, 15.5);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  document.body.prepend(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 8;
  controls.maxDistance = 27;
  controls.target.set(0, 0.2, 0);

  scene.add(new THREE.HemisphereLight(0x9bdcff, 0x071018, 1.1));
  const cyanLight = new THREE.PointLight(0x55dfff, 35, 25);
  cyanLight.position.set(-6, 5, 7);
  scene.add(cyanLight);
  const amberLight = new THREE.PointLight(0xff9944, 28, 22);
  amberLight.position.set(6, -2, 5);
  scene.add(amberLight);

  function makeSpacecraft(accent) {
    const craft = new THREE.Group();
    const metal = new THREE.MeshStandardMaterial({ color: 0x8797a6, metalness: 0.82, roughness: 0.27 });
    const darkMetal = new THREE.MeshStandardMaterial({ color: 0x1a2731, metalness: 0.72, roughness: 0.34 });
    const glow = new THREE.MeshStandardMaterial({ color: accent, emissive: accent, emissiveIntensity: 1.4, metalness: 0.35, roughness: 0.24 });
    const solar = new THREE.MeshStandardMaterial({ color: 0x123e68, emissive: 0x071b36, emissiveIntensity: 0.8, metalness: 0.5, roughness: 0.24 });

    const bus = new THREE.Mesh(new THREE.BoxGeometry(2.0, 1.25, 1.35), metal);
    craft.add(bus);

    const face = new THREE.Mesh(new THREE.BoxGeometry(1.55, 0.82, 0.04), darkMetal);
    face.position.z = 0.696;
    craft.add(face);

    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.34, 0.52, 24), glow);
    lens.rotation.x = Math.PI / 2;
    lens.position.set(0, 0.05, 0.94);
    craft.add(lens);

    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 1.0, 12), metal);
    mast.position.y = 1.08;
    craft.add(mast);
    const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.48, 0.18, 28), metal);
    dish.position.y = 1.62;
    craft.add(dish);

    for (const side of [-1, 1]) {
      const boom = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.08, 0.08), metal);
      boom.position.x = 1.45 * side;
      craft.add(boom);
      const panel = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.06, 1.0), solar);
      panel.position.x = 2.95 * side;
      craft.add(panel);
      for (let line = -2; line <= 2; line += 1) {
        const cellLine = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.075, 0.96), glow);
        cellLine.position.set(2.95 * side + line * 0.39, 0.04, 0);
        craft.add(cellLine);
      }
    }

    for (const x of [-0.62, 0.62]) {
      const thruster = new THREE.Mesh(new THREE.ConeGeometry(0.19, 0.43, 16), darkMetal);
      thruster.rotation.x = -Math.PI / 2;
      thruster.position.set(x, -0.36, -0.86);
      craft.add(thruster);
    }

    const axes = new THREE.AxesHelper(2.4);
    axes.material.transparent = true;
    axes.material.opacity = 0.82;
    craft.add(axes);
    craft.scale.setScalar(0.72);
    return craft;
  }

  const craftA = makeSpacecraft(0x56dfff);
  const craftB = makeSpacecraft(0xff9944);
  craftA.position.x = -4.0;
  craftB.position.x = 4.0;
  scene.add(craftA, craftB);

  const grid = new THREE.GridHelper(34, 34, 0x18455d, 0x0b2638);
  grid.position.y = -2.8;
  grid.material.transparent = true;
  grid.material.opacity = 0.42;
  scene.add(grid);

  const starCount = 1700;
  const starPositions = new Float32Array(starCount * 3);
  for (let index = 0; index < starCount; index += 1) {
    const radius = 35 + Math.random() * 110;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    starPositions[index * 3] = radius * Math.sin(phi) * Math.cos(theta);
    starPositions[index * 3 + 1] = radius * Math.cos(phi);
    starPositions[index * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
  }
  const starGeometry = new THREE.BufferGeometry();
  starGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  scene.add(new THREE.Points(starGeometry, new THREE.PointsMaterial({ color: 0xb9e8ff, size: 0.11, transparent: true, opacity: 0.78 })));

  const ui = {
    antipode: document.querySelector("#mode-antipode"),
    mismatch: document.querySelector("#mode-mismatch"),
    mismatchKind: document.querySelector("#mismatch-kind"),
    play: document.querySelector("#play"),
    axis: document.querySelector("#axis"),
    angle: document.querySelector("#angle"),
    angleOutput: document.querySelector("#angle-output"),
    error: document.querySelector("#attitude-error"),
    qA: document.querySelector("#q-a"),
    qB: document.querySelector("#q-b"),
    labelA: document.querySelector("#label-a-detail"),
    labelB: document.querySelector("#label-b-detail"),
    explanation: document.querySelector("#explanation"),
  };

  let mode = "antipode";
  let playing = true;
  let angleDegrees = 0;

  function selectedAxis() {
    if (ui.axis.value === "x") return new THREE.Vector3(1, 0, 0);
    if (ui.axis.value === "y") return new THREE.Vector3(0, 1, 0);
    if (ui.axis.value === "123") return new THREE.Vector3(1, 2, 3).normalize();
    return new THREE.Vector3(0, 0, 1);
  }

  function formatNumber(value) {
    const normalized = Math.abs(value) < 0.0005 ? 0 : value;
    return `${normalized >= 0 ? "+" : "−"}${Math.abs(normalized).toFixed(3)}`;
  }

  function formatQuaternion(q) {
    return `(${formatNumber(q.w)}, ${formatNumber(q.x)}, ${formatNumber(q.y)}, ${formatNumber(q.z)})`;
  }

  function physicalErrorDegrees(lhs, rhs) {
    const dot = Math.abs(lhs.dot(rhs));
    return THREE.MathUtils.radToDeg(2 * Math.acos(THREE.MathUtils.clamp(dot, 0, 1)));
  }

  function updateMode(nextMode) {
    mode = nextMode;
    ui.antipode.classList.toggle("active", mode === "antipode");
    ui.mismatch.classList.toggle("active", mode === "mismatch");
    ui.mismatchKind.disabled = mode !== "mismatch";
    renderAttitudes();
  }

  function renderAttitudes() {
    const q = new THREE.Quaternion().setFromAxisAngle(selectedAxis(), THREE.MathUtils.degToRad(angleDegrees)).normalize();
    let comparison;

    if (mode === "antipode") {
      comparison = new THREE.Quaternion(-q.x, -q.y, -q.z, -q.w);
      ui.labelA.textContent = "q";
      ui.labelB.textContent = "−q";
      ui.explanation.innerHTML = "<strong>Same physical attitude.</strong> SU(2) double-covers SO(3), so antipodal points q and −q produce the same rotation matrix. The sign is not a second engineering convention.";
    } else if (ui.mismatchKind.value === "layout") {
      // Correct wire bytes [x,y,z,w], incorrectly consumed by a scalar-first
      // constructor Quaternion{w,x,y,z}. THREE's constructor order is x,y,z,w.
      comparison = new THREE.Quaternion(q.y, q.z, q.w, q.x).normalize();
      ui.labelA.textContent = "[w,x,y,z] / correct";
      ui.labelB.textContent = "[x,y,z,w] / misread";
      ui.explanation.innerHTML = "<strong>Interface-contract failure.</strong> The same four numbers are valid, but interpreting [x,y,z,w] as [w,x,y,z] creates a different physical attitude. Named adapters and basis-vector tests catch this at the boundary.";
    } else {
      comparison = q.clone().conjugate();
      ui.labelA.textContent = "active body→world";
      ui.labelB.textContent = "passive/world→body";
      ui.explanation.innerHTML = "<strong>Action-direction failure.</strong> A passive coordinate transform is the inverse of the active vector rotation here, so silently crossing that boundary conjugates the quaternion and reverses the rotation.";
    }

    craftA.quaternion.copy(q);
    craftB.quaternion.copy(comparison);
    ui.qA.textContent = formatQuaternion(q);
    ui.qB.textContent = formatQuaternion(comparison);
    ui.angle.value = angleDegrees.toFixed(1);
    ui.angleOutput.textContent = `${angleDegrees.toFixed(1)}°`;
    const error = physicalErrorDegrees(q, comparison);
    ui.error.textContent = `${error.toFixed(3)}°`;
    ui.error.style.color = error < 0.001 ? "var(--cyan)" : "var(--orange)";
  }

  ui.antipode.addEventListener("click", () => updateMode("antipode"));
  ui.mismatch.addEventListener("click", () => updateMode("mismatch"));
  ui.mismatchKind.addEventListener("change", renderAttitudes);
  ui.axis.addEventListener("change", renderAttitudes);
  ui.angle.addEventListener("input", () => {
    angleDegrees = Number(ui.angle.value);
    playing = false;
    ui.play.textContent = "Play";
    renderAttitudes();
  });
  ui.play.addEventListener("click", () => {
    playing = !playing;
    ui.play.textContent = playing ? "Pause" : "Play";
  });
  addEventListener("keydown", (event) => {
    if (event.code === "Space") {
      event.preventDefault();
      ui.play.click();
    }
  });
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  const query = new URLSearchParams(location.search);
  const requestedAngle = Number(query.get("angle"));
  if (Number.isFinite(requestedAngle) && query.has("angle")) {
    angleDegrees = THREE.MathUtils.clamp(requestedAngle, 0, 720);
    playing = false;
    ui.play.textContent = "Play";
  }
  if (["layout", "passive"].includes(query.get("mismatch"))) {
    ui.mismatchKind.value = query.get("mismatch");
  }
  if (query.get("mode") === "mismatch") {
    mode = "mismatch";
    ui.antipode.classList.remove("active");
    ui.mismatch.classList.add("active");
    ui.mismatchKind.disabled = false;
  }

  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const delta = Math.min(clock.getDelta(), 0.05);
    if (playing) {
      angleDegrees = (angleDegrees + delta * 34) % 720;
      renderAttitudes();
    }
    controls.update();
    renderer.render(scene, camera);
  }

  renderAttitudes();
  animate();
})();
