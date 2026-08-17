use quaternion_convention::{Quaternion, Vector3};

fn main() {
    let q = Quaternion::from_axis_angle(
        Vector3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        std::f64::consts::FRAC_PI_2,
    );
    let antipode = q.negated();
    println!("Contract: Hamilton, scalar-first, active body-to-world");
    println!(
        "q versus -q physical error: {:.6} deg",
        q.physical_rotation_distance(antipode).to_degrees()
    );

    let wire = q.to_scalar_last();
    let misread = Quaternion {
        w: wire.x,
        x: wire.y,
        y: wire.z,
        z: wire.w,
    }
    .normalized();
    println!(
        "scalar-last bytes read scalar-first: {:.6} deg error",
        q.physical_rotation_distance(misread).to_degrees()
    );

    let aligned = q.align_hemisphere(antipode);
    println!("hemisphere-aligned dot product: {:.6}", q.dot(aligned));
}
