# Prompt-following eval — `ckpt/instruct_v4` (step 2000)

200 held-out prompts, T=0.8 top-p=0.95, seed 1234.

| prompt kind | n | follows |
|---|---|---|
| begins | 39 | 87.2% |
| words | 161 | 27.3% |
| **all** | 200 | **39.0%** |

ended with EOT: 196/200; mean rep4 0.023; requested words present (words prompts): 62.7%

## First examples

**Prompt: Write a short story that uses the words "crown", "king" and "mommy".
Story:** — follows: False

> Once upon a time, there was a big king. He had a shiny crown on his head. The king was very bossy. He told the people in his land to be quiet. The people did not like this, but they were too scared to go outside.
> One day, a little girl went to the king's castle. She knocked on the door and said, "King, please shut the door." The king did not want to shut the door. He said, "No, I will not shut the door."
> The king did not listen to the little girl. He shut the door very fast. The little girl was very sad. She went outside to get the king. She was scared that the king would be angry with her.
> The king was in a lot of trouble. He could not find the little girl and the big castle. The king was very angry and the people were very sad. They did not like the bossy king anymore.<|endoftext|>

**Prompt: Write a short story that uses the words "behind", "park" and "played".
Story:** — follows: False

> Once upon a time, there was a nosy cat named Kitty. Kitty loved to play with her ball and chase butterflies. One day, Kitty saw a big bush with a lot of red berries. She wanted to play a game.
> Kitty asked her friend, Doggy, "What game should we play?" Doggy thought for a moment and said, "Let's play hide and seek!" Kitty liked the idea and wagged her tail.
> Kitty and Doggy played hide and seek. Kitty hid behind the bush, and Doggy looked for her. But then, Kitty jumped out from behind the bush. "Surprise!" she yelled, laughing. Doggy laughed too and they played more games together.<|endoftext|>

**Prompt: Write a short story that uses the words "named", "play" and "warm".
Story:** — follows: True

> Once upon a time, there was a little dog named Max. Max was a cold dog. He liked to play outside when the sun was out. Max loved to run and jump in the warm sun.
> One day, Max saw a big cat named Tom. Tom was playing in the warm sun. Max wanted to play too, but Tom did not want to play with Max. Max was sad.
> Max tried to play with Tom, but Tom did not want to play. Tom got mad at Max. Max was sad and cold. Max went home and wished he had a warm bed.<|endoftext|>

**Prompt: Write a short story that uses the words "body", "friends" and "home".
Story:** — follows: False

> One day, a bald man named Tom went for a walk. He saw a little bird on the ground. The bird could not fly. Tom said, "Hi, bird! What is your name?"
> The bird said, "My name is Billy. I cannot fly like you."
> Tom wanted to help Billy. He had an idea. He said, "I will order a big friend to help you fly."
> Tom went to his friend's house. He said, "A big friend is a bird who can help me fly."
> His friend said, "I have a big friend who can help you." The bird told them about a big bird who could help Billy. The big bird came and helped Billy. Billy flew high in the sky.
> Billy was happy to fly. Tom and his friends were happy too. They all played together and had fun.<|endoftext|>

