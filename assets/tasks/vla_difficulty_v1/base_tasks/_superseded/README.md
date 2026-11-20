Superseded drafts (kept for history, not compiled):

- T02_mug_to_table.yaml — goal breakfast_table_uhrsex_0: PLACE collides from every
  reachable base pose (chairs around the table); S3 dining room behind a 0.75 m door.
- T04_apple_to_living_room.yaml — goal coffee_table_qlmqyy_0: PLACE collides (low table);
  no living-room surface was PLACE-feasible.

Evidence: runs/vla_difficulty_v1/_locations/round3_mug.json, round4_goals_mug.json.
Replacements: T02_mug_to_counter.yaml, T04_apple_to_island.yaml.
- T04_apple_to_island.yaml — goal countertop_tpuwys_4: whether the apple could be PLACEd
  depended on how it had been grasped (from countertop_tpuwys_0 and from the far end of
  bookcase_owvfik_1: COLLISION at every pose, including poses that worked after other grasps).
- T03_apple_to_fridge.yaml — inside(apple, fridge): PLACE had no inside support for fixed
  furniture, and no collision-free base pose exists in front of the open fridge.
Replacements: T04_apple_to_counter.yaml (goal certified for grasps from all three S
locations with the mug), T03_fork_to_drawer.yaml (inside-drawer PLACE added 2026-10-18).
