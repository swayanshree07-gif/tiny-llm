# Sample suite — checkpoint step 27495

6 prompts x 4 settings, max 240 new tokens, seed 1234. `rep4` = fraction of repeated 4-grams (lower is better), `distinct2` = unique bigrams / bigrams, `ended` = model emitted `<|endoftext|>`.

## Summary by setting

| setting | mean words | ended | mean distinct2 | mean rep4 |
|---|---|---|---|---|
| greedy | 145 | 5/6 | 0.734 | 0.069 |
| T=0.7 top-k=40 | 119 | 6/6 | 0.807 | 0.052 |
| T=0.8 top-p=0.95 | 133 | 5/6 | 0.771 | 0.072 |
| T=1.0 top-p=0.9 | 149 | 5/6 | 0.842 | 0.019 |

## Samples

### `Once upon a time` — greedy
*{'words': 165, 'ended': True, 'distinct2': 0.848, 'rep4': 0.025}*

Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around the house. One day, she saw a big, red ball in the living room. She wanted to play with it, but it was too high for her to reach.

Lily had an idea. She found a chair and pushed it to the couch. She climbed up the chair and reached for the ball. But, oh no! The chair was too heavy for her. She tried and tried, but she could not move it.

Then, her big brother Tom came in. He saw Lily trying to get the ball. Tom wanted to help, so he picked up the chair and moved it to the couch. Lily was so happy! She thanked Tom and they played with the big, red ball together.

### `Once upon a time` — T=0.7 top-k=40
*{'words': 196, 'ended': True, 'distinct2': 0.856, 'rep4': 0.005}*

Once upon a time, there was a little girl named Mia. She loved to play outside and look at the pretty flowers. One day, Mia went to the park with her mom. She was very happy.

While playing at the park, Mia saw a small, harmless bug on a leaf. She wanted to look at it more, so she went closer. The bug looked at Mia and said, "Hi, my name is Tim. I am a friendly bug. What is your name?" Mia smiled and said, "Hello, Tim! My name is Mia. Nice to meet you!"

Mia and Tim played together in the park. They played with a ball and ran around. They had so much fun. After a while, they both felt tired. Mia said, "I am tired, Tim. Let's go home." Tim said, "Goodbye, Mia. See you soon!" And they both went home, excited to play again the next day.

### `Once upon a time` — T=0.8 top-p=0.95
*{'words': 153, 'ended': True, 'distinct2': 0.842, 'rep4': 0.0}*

Once upon a time, there was a little girl named Amy. Amy loved to play in the park. One day, she saw a big slide. She was so happy! She ran to the slide and climbed up.

Amy got ready to slide down. She said, "I will, Mom!" She slid down the slide very fast. Her mom watched her and smiled.

But then, something unexpected happened. A little bird flew by and saw Amy on the slide. The bird thought the slide was a nest. The bird flew to the top of the slide and started to sing a song. Amy saw the bird and smiled too. They both climbed up the slide together. The bird and Amy became best friends, and they played on the slide every day.

### `Once upon a time` — T=1.0 top-p=0.9
*{'words': 154, 'ended': True, 'distinct2': 0.81, 'rep4': 0.013}*

Once upon a time, there was a little boy named Tim. Tim had a penny. He loved his penny very much. One day, he saw a big rock. He wanted to see how big it was. So, Tim started to kneel on the rock.

A terrible wind came and made the rock move. The rock was so big that it covered Tim's house. Tim was very sad. He did not like the big rock. He wanted his house to be nice again.

Then, a kind bird saw Tim. The bird wanted to help. The bird picked up the big rock with its beak. The bird carried the rock away from the house. The bird made a new house for Tim. Tim was happy again. He thanked the bird and they became friends.

### `Lily and her dog went to the park.` — greedy
*{'words': 98, 'ended': True, 'distinct2': 0.753, 'rep4': 0.011}*

Lily and her dog went to the park. They saw a big tree. Lily wanted to climb the tree. She started to climb up. Her dog barked, "Woof! Woof!" Lily was happy.

But then, Lily saw a big, red ball. She wanted to play with it. She forgot about the tree. She ran to the ball and kicked it. The ball went up, up, up. Lily was sad. She could not get the ball. The end.

### `Lily and her dog went to the park.` — T=0.7 top-k=40
*{'words': 101, 'ended': True, 'distinct2': 0.73, 'rep4': 0.102}*

Lily and her dog went to the park. They saw a big tree. Lily wanted to climb the tree. Her dog barked, "Woof, woof!" Lily started to climb. She climbed higher and higher. Her dog barked, "Woof, woof!" Lily felt happy.

When they were at the top of the tree, they saw a bird. The bird was singing a beautiful song. Lily and her dog listened to the bird's song. They smiled and had a fun day at the park.

### `Lily and her dog went to the park.` — T=0.8 top-p=0.95
*{'words': 84, 'ended': True, 'distinct2': 0.88, 'rep4': 0.0}*

Lily and her dog went to the park. They saw a big, pretty butterfly. The butterfly was very colorful. It had many colors. Lily wanted to show the butterfly to her mom.

Lily said to the butterfly, "Look, Mommy! A pretty butterfly!" The butterfly came closer to Lily. She was so happy. The butterfly and Lily played in the park all day. They had lots of fun together.

### `Lily and her dog went to the park.` — T=1.0 top-p=0.9
*{'words': 246, 'ended': False, 'distinct2': 0.665, 'rep4': 0.041}*

Lily and her dog went to the park. They wanted to play catch. Lily threw the ball and the dog ran to catch it. They were very happy.

But then, they saw a boy. The boy was rude. He said mean things to the dog. He said the dog was bad too. The dog did not care. He said, "No, go away. You are not good. You are not nice. You are mean."

Lily was sad. She did not like the boy. He was rude. She did not want to play with him. She liked her yellow ball. She threw it away. The dog had fun. He was not rude.

Then, something happened. The boy came and took Lily's ball. He ran away with it. He hid behind a tree. He said, "No, no, no. I will not give you my ball. You will never play with me."

Lily was surprised. She did not expect the boy to take her ball. She thought he was mean. She said, "Please, okay. I will not get my ball. I will just play with the dog. He is nice. Please, let me keep my ball."

The boy

### `Tom was sad because` — greedy
*{'words': 114, 'ended': True, 'distinct2': 0.867, 'rep4': 0.0}*

Tom was sad because he could not play with the ball. He sat down and thought about how to get the ball. Then, he had an idea. He would ask his friend, the dog named Max, for help.

Tom went to Max and asked, "Max, can you help me get the ball?" Max wagged his tail and said, "Yes, Tom! I will help you." Max stood up and used his nose to push the ball out of the tree. Tom was so happy! He thanked Max and they played with the ball together.

### `Tom was sad because` — T=0.7 top-k=40
*{'words': 99, 'ended': True, 'distinct2': 0.929, 'rep4': 0.0}*

Tom was sad because he had no friends to play with. He looked around and saw a little girl named Lily. Lily was playing with a ball. Tom went to Lily and asked, "Can I play with you?"

Lily smiled and said, "Yes, you can play with me!" They played with the ball and became friends. Tom was happy that he had a new friend. From that day on, Tom and Lily played together every day. They were the best of friends.

### `Tom was sad because` — T=0.8 top-p=0.95
*{'words': 104, 'ended': True, 'distinct2': 0.816, 'rep4': 0.01}*

Tom was sad because he could not find his ball. He looked everywhere, but he could not find it. Then, he saw a big dog. The dog had his ball! Tom was scared, but he wanted his ball back. He thought of a plan.

Tom said to the dog, "Can we have my ball back, please?" The dog was nice and gave the ball to Tom. Tom was very happy. He played with the dog and his ball. Now, Tom and the dog were friends.

### `Tom was sad because` — T=1.0 top-p=0.9
*{'words': 114, 'ended': True, 'distinct2': 0.912, 'rep4': 0.0}*

Tom was sad because he did not mean to make his friend suffer. He wanted to make her feel better. So, he thought of a plan. Tom and Max decided to clean the house together. They picked up toys, books, and food. They put everything away and made the house clean again.

When they were done, Tom and Max looked at the house and smiled. They were happy that they fixed it. Now, they could play and be friends again. The moral of the story is that when you help your friends, you can make them happy again.

### `One day, a little bird` — greedy
*{'words': 159, 'ended': True, 'distinct2': 0.772, 'rep4': 0.038}*

One day, a little bird named Tim was flying in the sky. He saw a big tree and wanted to rest. Tim flew down and sat on a branch. He looked around and saw a big, red ball on the ground. Tim was very happy to find the ball.

Tim flew down to the ground and picked up the ball with his beak. He flew back up to the tree and dropped the ball down. Tim picked up the ball with his beak and started to play with it. He was having so much fun.

After a while, Tim's friend, a little squirrel named Sam, came to the tree. Sam saw Tim with the ball and wanted to play too. They both played with the ball and had a great time. They were very happy to have each other as friends.

### `One day, a little bird` — T=0.7 top-k=40
*{'words': 131, 'ended': True, 'distinct2': 0.9, 'rep4': 0.0}*

One day, a little bird named Tim wanted to play a game. He wanted to see who could fly the highest in the sky. He was very excited.

Tim flapped his wings and started to fly. But, oh no! He could not fly as high as his friends. He was sad and tired. Tim sat on a tree branch and cried.

A kind squirrel saw Tim and helped him. The squirrel told Tim to keep trying and not to give up. Tim listened to the squirrel and kept trying. He flapped his wings harder and harder. Finally, Tim flew higher than ever before. He was so happy and thanked the squirrel for his help.

### `One day, a little bird` — T=0.8 top-p=0.95
*{'words': 146, 'ended': True, 'distinct2': 0.745, 'rep4': 0.028}*

One day, a little bird was in a tree. The bird was sad because it could not find its nest. The bird began to wonder where its nest was. It wanted to look around and see if it could find it.

The bird flew around and saw a big tree. It looked in the tree and saw many eggs. The bird was happy because it knew it would find its nest. But then, the egg started to shake! The bird was scared and surprised.

As the egg cracked open, a baby bird came out of the egg. The baby bird was happy and safe. The bird was surprised too. The bird and the baby bird became good friends. They lived together in the tree and had lots of fun.

### `One day, a little bird` — T=1.0 top-p=0.9
*{'words': 207, 'ended': True, 'distinct2': 0.84, 'rep4': 0.01}*

One day, a little bird came to visit the garden. The bird was very happy to meet the farmer and live there. The bird would sing songs and look at the farmer, and the farmer would laugh and clap his hands.

The farmer had a pet bee named Buzz. Buzz was a small, green bug who lived in a jar in the garden. Buzz was a very intelligent bee and could always find answers to what was happening. He saw the birds and the bees, and he wanted to say hello.

One day, Buzz was trying to call to his friend, the bee. But he could not find him. Buzz was sad and did not know what to do. Then, he saw a butterfly flying by and shouted, "Frow a bee!"

The butterfly heard Buzz's shout and flew away fast. Buzz was so happy to see the butterfly. He shouted back, "Free bee! Five hive!" The butterfly came back and buzzed around Buzz. They were together again, and Buzz and the butterfly were now best friends.

### `The big red ball` — greedy
*{'words': 94, 'ended': True, 'distinct2': 0.624, 'rep4': 0.088}*

The big red ball was sad. It wanted to play with the kids. The kids saw the ball and wanted to help. They had an idea. They would give the ball a big hug. The big red ball liked the hug. It felt happy.

The big red ball and the kids hugged the ball. The ball was happy too. It was not sad anymore. The kids were happy too. They all played together. The big red ball was not sad anymore.

### `The big red ball` — T=0.7 top-k=40
*{'words': 81, 'ended': True, 'distinct2': 0.675, 'rep4': 0.115}*

The big red ball wanted to play with the little blue ball. But the little blue ball did not want to play with the big red ball. The big red ball was sad.

Then, the big red ball had an idea. It said, "Let's play together!" The little blue ball was happy. They played and played. The big red ball was not sad anymore. They became good friends.

### `The big red ball` — T=0.8 top-p=0.95
*{'words': 70, 'ended': True, 'distinct2': 0.826, 'rep4': 0.045}*

The big red ball came out of the box and rolled away very fast. It went into the street and got a big road. The big red ball tried to drive fast, but it was too late. The street had a big street with many cars. The big red ball could not get to the store. The little red ball was very sad.

### `The big red ball` — T=1.0 top-p=0.9
*{'words': 82, 'ended': True, 'distinct2': 0.951, 'rep4': 0.013}*

The big red ball rolled off the track. It was rolling down the hill! Tim ran after it, trying to catch it. When he got to the top, he saw a big, red ball. The big red ball could talk! It said, "Hello, Tim! Let's play together!" Tim was so surprised! He had never met a talking ball before. They played together all day long.

### `Mom said, "` — greedy
*{'words': 241, 'ended': False, 'distinct2': 0.542, 'rep4': 0.252}*

Mom said, "Yes, you can. But be careful. Don't touch the hot stove. It can burn you."

Lily and Ben nodded. They took the pot and the spoon and went to the living room. They put the pot on the stove and turned it on. They saw the water and the bubbles. They smelled the spicy food. They were curious.

They took turns to stir the spicy food. They made a lot of noise. They did not hear Mom coming. They did not see the pot on the stove. They did not see the water on the floor. They did not see the water on the stove.

Mom saw the pot on the stove. She saw the water on the floor. She saw Lily and Ben. She was angry. She said, "What are you doing? You could have burned yourself! You could have burned yourself! You could have burned yourself! You are in big trouble!"

Lily and Ben were scared. They did not mean to make Mom angry. They did not mean to make her mad. They did not mean to make a mess. They did not mean to make Mom angry. They did not mean to make her mad. They did not

### `Mom said, "` — T=0.7 top-k=40
*{'words': 105, 'ended': True, 'distinct2': 0.75, 'rep4': 0.088}*

Mom said, "I love you too, Lily. And I love the cake. It is a special cake. It is a cake for a party. We will have a party soon. But first, we have to clean up the mess. And we have to put the pieces back in the box. Can you do that?"

Lily nodded. She put the pieces back in the box. She helped her mom to pick up the pieces. She was a good helper. She was a good girl. She was happy.

### `Mom said, "` — T=0.8 top-p=0.95
*{'words': 242, 'ended': False, 'distinct2': 0.519, 'rep4': 0.347}*

Mom said, "Yes, Tom. We can be careful and not break anything."

They went to the kitchen and opened the fridge. They saw some cheese, some ham, some eggs, some cheese, some milk, some bread, some cheese, some cheese, some cheese, some ham, some eggs, some cheese, some cheese, some bread, some cheese, some ham, some cheese, some ham, some ham, some ham, some cheese, some ham, some cheese, some ham, some ham, some ham, some ham, some ham, some ham, some ham, some ham, some ham, some cheese. They make their own home and get some ham and some ham, too. They make a door, a window, a chimney, a chimney, a chimney, a chimney, a door, and a chimney. They are very proud of their home and their ham suddenly, they hear a noise outside.

"What is that?" Tom asks.

"I think it is a bird or a bird. Maybe they are making a cocoon or a trap," Mom says.

Tom and Sam are very curious and excited. They want to see the bird and the nest.

### `Mom said, "` — T=1.0 top-p=0.9
*{'words': 89, 'ended': True, 'distinct2': 0.875, 'rep4': 0.035}*

Mom said, "Yes, you can help me. Let's count the fruits together." They started counting and counting.

But then, a big wind came. It blew the big fruit away. Lily and her mom were sad. They looked for the fruit, but they could not find it. The wind had taken it far away.

Lily and her mom went home. They were sad because they did not find the big fruit. The end.

