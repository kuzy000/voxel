use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};

const FONT_SIZE: f32 = 18.;

pub struct GameUiPlugin;

impl Plugin for GameUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        app.add_systems(Update, update_fps);
    }
}

#[derive(Component)]
pub struct TextFps;

pub fn update_fps(diag: Res<DiagnosticsStore>, mut q: Query<&mut Text, With<TextFps>>) {
    let mut text_fps = q.single_mut();

    let Some(fps) = diag.get_measurement(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };

    *text_fps = Text::from_section(
        format!("FPS: {:.2}", fps.value),
        TextStyle {
            font_size: FONT_SIZE,
            color: Color::rgb(0.9, 0.9, 0.9),
            ..default()
        },
    );
}

pub fn setup(mut commands: Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.),
                height: Val::Percent(100.0),
                align_items: AlignItems::FlexStart,
                justify_content: JustifyContent::Stretch,
                flex_direction: FlexDirection::Column,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(250.),
                        margin: UiRect::bottom(Val::Px(15.)),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|parent| {
                    parent
                        .spawn(TextBundle {
                            text: Text::from_section(
                                "",
                                TextStyle {
                                    font_size: FONT_SIZE,
                                    color: Color::rgb(0.9, 0.9, 0.9),
                                    ..default()
                                },
                            ),
                            ..default()
                        })
                        .insert(TextFps);
                });
        });
}
