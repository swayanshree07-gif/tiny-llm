# Scripted chat session — ckpt/chat_large_v3 (2026-11-09)

Command: `printf ... | scripts/chat.py --ckpt ckpt/chat_large_v3 --seed 7 --max-turns 4` (T=0.8, top-p 0.95, KV cache).
Stderr notes `[dropped …]` when old turns were removed to fit the 256-token context.

```
User: Hello!
Assistant: Hello! I can write short stories for you. Give me some words or a first sentence.

[dropped 1 old turn(s) to fit the context]
User: Write a short story that uses the words "dog", "ball" and "park".
Assistant: One day, a little boy named Tim went to the park with his mom. They saw a big red ball on the grass. Tim was very happy and wanted to play with it. His mom said, "Tim, be careful not to throw the ball too hard. If it goes too far, it might go into the woods."
Tim played with the ball for a while. He threw it up and down, and he laughed. Then, he saw a tree with a knot in it. Tim thought it was a very funny tree. He wanted to see what was up there.
He found a long rope and tried to pull it down. But the knot was too tight. Tim's mom saw him and said, "Tim, be careful! Don't drop the ball. It might go into the tree." Tim listened to his mom and played more carefully. He threw the ball near the tree, and it went into the knot. Tim was very happy and proud of himself.

[dropped 1 old turn(s) to fit the context]
User: Continue this story: "Lily found a small blue box under her bed."
Assistant: Lily found a small blue box under her bed. She opened the box and saw a tiny crown inside. It was shiny and pretty. Lily put the crown on her head and felt very happy.
Lily's friend, Tom, came over to play. He saw the crown and said, "Wow, I like your crown, Lily!" Lily smiled and said, "Thank you, Tom. Do you want to play a game with me?" Tom was excited and said, "Yes, I want to play!"
They played a game where they had to balance on one foot. They took turns and laughed when they were tired. After playing, they put the crown back in the box and went home. They couldn't wait to play again tomorrow.

[dropped 1 old turn(s) to fit the context]
User: Thank you!
Assistant: You are welcome! Do you want another story?

```
