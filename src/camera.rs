use bevy::{input::mouse::MouseMotion, prelude::*, render::{extract_component::ExtractComponent, primitives::Frustum}};

#[derive(Component, Clone, ExtractComponent)]
pub struct GameCamera;

#[derive(Bundle)]
pub struct GameCameraBundle {
    pub marker: GameCamera,
    pub projection: Projection,
    pub frustum: Frustum,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
}

impl Default for GameCameraBundle {
    fn default() -> Self {
        Self {
            marker: GameCamera,
            projection: Default::default(),
            frustum: Default::default(),
            transform: Default::default(),
            global_transform: Default::default(),
        }
    }
}

pub fn update_game_camera(
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion_evr: EventReader<MouseMotion>,
    mut q: Query<(&mut Transform, &mut Projection), With<GameCamera>>,
) {
    let (mut transform, mut projection) = q.single_mut();

    if let Projection::Perspective(ref mut p) = *projection {
        p.aspect_ratio = 1280. / 720.;
    }

    let speed = if input.pressed(KeyCode::ShiftLeft) {
        500.
    } else {
        100.
    };

    let mut v = Vec3::ZERO;

    let forward = -Vec3::from(transform.local_z());
    let right = Vec3::from(transform.local_x());
    let up = Vec3::from(transform.local_y());

    if input.pressed(KeyCode::KeyW) {
        v += forward;
    }

    if input.pressed(KeyCode::KeyS) {
        v -= forward;
    }

    if input.pressed(KeyCode::KeyA) {
        v -= right;
    }

    if input.pressed(KeyCode::KeyD) {
        v += right;
    }

    if input.pressed(KeyCode::KeyQ) {
        v -= up;
    }

    if input.pressed(KeyCode::KeyE) {
        v += up;
    }

    let mut x = 0.;
    let mut y = 0.;

    if buttons.pressed(MouseButton::Right) {
        for ev in motion_evr.read() {
            x += ev.delta.x;
            y += ev.delta.y;
        }
    }

    let factor = 0.01;

    transform.translation += v.normalize_or_zero() * time.delta_seconds() * speed;

    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);

    let yaw = yaw - x * factor;
    let pitch = pitch - y * factor;

    transform.rotation = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);
}