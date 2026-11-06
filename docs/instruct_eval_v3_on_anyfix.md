# Prompt-following eval — `ckpt/instruct_v3` (step 2000)

200 held-out prompts, T=0.8 top-p=0.95, seed 1234.

| prompt kind | n | follows |
|---|---|---|
| begins | 35 | 82.9% |
| words | 165 | 2.4% |
| **all** | 200 | **16.5%** |

ended with EOT: 195/200; mean rep4 0.021; requested words present (words prompts): 30.9%

## First examples

**Prompt: Write a short story that uses the words "crown", "eyes" and "needed".
Story:** — follows: False

> Once upon a time, there was a gifted dog named Spot. Spot could do many things that other dogs could not do. One day, Spot went for a walk with his friend, a little girl named Lily.
> As they walked, Lily and Spot saw a big red ball. Lily wanted to play with the ball, but she forgot what her mom told her. "Spot, we need to find the ball," she said. Spot wagged his tail and ran to the ball.
> Spot picked up the ball with his mouth and brought it back to Lily. She was very happy and threw the ball for Spot. They played with the ball all day and had lots of fun. And from that day on, Spot always remembered to bring the ball with him when they played.<|endoftext|>

**Prompt: Write a short story that uses the words "behind", "seek" and "walked".
Story:** — follows: False

> One day, a little boy named Tim went to play hide and seek with his friends. He was very excited to find the perfect hiding spot. He ran to the big tree and hid behind it. "I will be the best hiding spot!" he said to his friends.
> Tim's friends looked and looked for him. They looked behind the bushes and under the bench. Finally, they found Tim hiding in the bushes. "Got you!" they shouted. Tim laughed and jumped out from his hiding spot.
> "You found me!" Tim said to his friends. They all laughed and played more hide and seek. After a while, Tim's friend, Sally, said, "I have a treat for you!" She pulled out a big, sweet ice cream cone.
> Tim and his friends sat down to eat the ice cream. But when they took a bite, they saw it was not ice cream. It was a red ball! They laughed and played with the red ball. They had so much fun playing with it that they forgot all about the game.<|endoftext|>

**Prompt: Write a short story that uses the words "shapes", "started" and "triangle".
Story:** — follows: False

> Once upon a time, there was a little boy named Tim. Tim was a nervous boy who did not like to try new things. One day, his mom told him they were going to a big party. Tim was scared but his mom said, "It will be fun, I promise."
> At the party, Tim saw a big triangle toy. He wanted to play with it, but he was nervous. He thought about how he might make friends with the other kids. Tim took a deep breath and walked up to the triangle toy.
> Just then, a girl named Sue came up to him. She was not nervous anymore. Sue said, "Hi, I'm Sue. Do you want to play with me?" Tim smiled and said, "Yes, let's play together!" They played with the triangle toy and had a lot of fun. Tim was not nervous anymore, and he made a new friend at the party.<|endoftext|>

**Prompt: Write a short story that uses the words "great", "outside" and "used".
Story:** — follows: False

> Once upon a time, there was a boy named Tom. Tom liked to play outside. One day, Tom saw a big, modern house. He thought it was fun to play in the house.
> Tom went inside the house and saw a faucet. He turned the faucet on and water came out. He liked the water. Tom played in the water and got wet. He had lots of fun.
> Tom heard his mom call him. "Tom, come here!" she said. Tom put down the water and went to his mom. She was happy to see him. She said, "You are a great wet boy, Tom!" They both laughed and played together all day.<|endoftext|>

