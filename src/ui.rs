use bevy::{diagnostic::DiagnosticsStore, prelude::*};

const FONT_SIZE: f32 = 18.;

pub struct GameUiPlugin;

impl Plugin for GameUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        app.add_systems(Update, update_fps);
    }
}

#[derive(Component)]
pub struct DiagContainer;

#[derive(Component)]
pub struct DiagChild;

pub fn update_fps(
    mut commands: Commands,
    diag: Res<DiagnosticsStore>,
    q_parent: Query<(Entity, Option<&Children>), With<DiagContainer>>,
    mut q_child: Query<&mut Text, With<DiagChild>>,
) {
    let mut diags: Vec<_> = diag.iter().collect();
    diags.sort_by(|a, b| a.path().as_str().cmp(b.path().as_str()));

    for (container_id, children) in q_parent.iter() {
        let children_len = children.map_or(0, |c| c.len());

        let mut idx = 0;
        for diag in &diags {
            let Some(value) = diag.smoothed() else {
                continue;
            };

            let text = Text::from_section(
                format!("{}: {:.2} {}", diag.path().as_str(), value, diag.suffix),
                TextStyle {
                    font_size: FONT_SIZE,
                    color: Color::rgb(0.9, 0.9, 0.9),
                    ..default()
                },
            );

            if idx < children_len {
                let children = children.unwrap();
                let mut text_mut = q_child.get_mut(children[idx]).unwrap();
                *text_mut = text;
            } else {
                commands.entity(container_id).with_children(|parent| {
                    parent
                        .spawn(TextBundle {
                            text: text,
                            ..default()
                        })
                        .insert(DiagChild);
                });
            }

            idx += 1;
        }

        if children_len > idx {
            let children = children.unwrap();
            let (_, b) = children.split_at(children.len());
            commands.entity(container_id).remove_children(b);
        }
    }
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
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    ..default()
                })
                .insert(DiagContainer);
        });
}
