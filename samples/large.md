# Sample suite — checkpoint step 18000

6 prompts x 4 settings, max 240 new tokens, seed 1234. `rep4` = fraction of repeated 4-grams (lower is better), `distinct2` = unique bigrams / bigrams, `ended` = model emitted `<|endoftext|>`.

## Summary by setting

| setting | mean words | ended | mean distinct2 | mean rep4 |
|---|---|---|---|---|
| greedy | 131 | 6/6 | 0.851 | 0.015 |
| T=0.7 top-k=40 | 106 | 6/6 | 0.865 | 0.019 |
| T=0.8 top-p=0.95 | 142 | 6/6 | 0.789 | 0.044 |
| T=1.0 top-p=0.9 | 92 | 6/6 | 0.853 | 0.024 |

## Samples

### `Once upon a time` — greedy
*{'words': 232, 'ended': True, 'distinct2': 0.827, 'rep4': 0.022}*

Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to play with. One day, she saw a small, green bug on the ground. Lily was not scared of the bug, so she wanted to be friends.

Lily said, "Hi, bug! Do you want to play with my ball?" The bug looked at her and said, "Yes, I want to play! But I am very hungry. Can you help me find some food?" Lily thought for a moment and then said, "I know! Let's go to the kitchen and find some food for you."

Lily and the bug went to the kitchen and found some yummy bugs for the bug to eat. The bug was very happy and said, "Thank you, Lily! You are very kind." Lily smiled and said, "You're welcome, bug! I'm glad I could help you."

From that day on, Lily and the bug became best friends. They played together every day and had lots of fun. And Lily always made sure to feed the bug, so they could play and have fun together.

### `Once upon a time` — T=0.7 top-k=40
*{'words': 158, 'ended': True, 'distinct2': 0.828, 'rep4': 0.032}*

Once upon a time, there was a little girl named Lily. She loved to play outside and look at the pretty flowers. One day, she saw a new flower in her garden. It was a tiny, delicate flower. Lily was very happy to see it.

Lily wanted the delicate flower to bloom, so she gave it water. Every day, she would talk to the flower and sing to it. The flower would listen to her voice and feel warm and happy. Lily knew that the flower would grow and bloom well.

As the days went by, the delicate flower grew big and strong. Lily would visit it every day and talk to it. The flower would listen and smile at her. It loved having Lily to talk to and touch it. And Lily was very happy, too.

### `Once upon a time` — T=0.8 top-p=0.95
*{'words': 170, 'ended': True, 'distinct2': 0.876, 'rep4': 0.03}*

Once upon a time, there was a little girl named Amy. Amy loved to play in the park. One day, she saw a big slide. Amy was so happy! She ran to the slide and climbed up.

As she was sliding, she saw a small cat stuck on the slide. The cat looked sad and scared. Amy wanted to help the cat. She tried to use a stick to help the cat come down. But the stick was not long enough.

Then, something unexpected happened. The cat started to talk! The cat said, "Thank you, Amy! I am a magic cat! Because you helped me, I will give you a special gift." The cat waved its tail, and suddenly, Amy's big, red ball rolled out from behind the slide. Amy was so happy and played with her ball all day long.

### `Once upon a time` — T=1.0 top-p=0.9
*{'words': 125, 'ended': True, 'distinct2': 0.895, 'rep4': 0.0}*

Once upon a time, there was a little dog named Max. Max had a shiny red collar. He loved to play and run in the park.

One day, Max was playing with his friend, a little girl named Lily. They were having so much fun. Lily said, "Max, let's go on the swings!" Max barked happily, and they ran to the swings.

But Max couldn't hear the birds outside. He was deaf. Lily was sad. She sat on a bench and hugged Max. Max looked at Lily and wagged his tail. They played on the swings and had a great day at the park.

### `Lily and her dog went to the park.` — greedy
*{'words': 105, 'ended': True, 'distinct2': 0.875, 'rep4': 0.01}*

Lily and her dog went to the park. They saw a big tree with a hole in it. Lily looked inside the hole and found a small, cute cat. The cat was scared and alone. Lily wanted to help the cat, so she took it home.

At home, Lily's mom helped her make a soft bed for the cat. They gave the cat some food and water. The cat was happy and not scared anymore. Lily and the cat became best friends. They played together every day and had lots of fun.

### `Lily and her dog went to the park.` — T=0.7 top-k=40
*{'words': 84, 'ended': True, 'distinct2': 0.904, 'rep4': 0.0}*

Lily and her dog went to the park. They saw a big tree with a swing. Lily sat on the swing and her dog sat on her lap. They started to swing back and forth, laughing and having fun.

"Look at me, Lily!" said Max. Lily laughed and did a little dance. They played on the swing until the sun went down. Then, they went home, happy and tired.

### `Lily and her dog went to the park.` — T=0.8 top-p=0.95
*{'words': 90, 'ended': True, 'distinct2': 0.775, 'rep4': 0.046}*

Lily and her dog went to the park. They played with the red ball. Lily threw the ball, and the dog ran to get it. They played for a long time. Lily was happy.

But then, the red ball went into the bushes. Lily and her dog looked for the ball. They found a big, shiny rock. The rock was very pretty. Lily and her dog liked the new rock. They played with it all day.

### `Lily and her dog went to the park.` — T=1.0 top-p=0.9
*{'words': 101, 'ended': True, 'distinct2': 0.83, 'rep4': 0.01}*

Lily and her dog went to the park. They wanted to play catch. Lily threw the ball and the dog ran to catch it. They were having so much fun. Lily's friend, Tom, came to the park too. Tom saw Lily and the dog playing.

"Can I play catch too?" Tom asked. Lily and the dog said, "Yes, let's play together!" So they played catch with the dog. They were all very happy. They had a great day at the park.

### `Tom was sad because` — greedy
*{'words': 86, 'ended': True, 'distinct2': 0.847, 'rep4': 0.0}*

Tom was sad because he wanted to play with the ball. He tried to get it, but the ball was too high. He jumped and jumped, but he could not reach it. Tom was very sad.

Then, a big dog named Max came to help Tom. Max was tall and could reach the ball. Max got the ball for Tom. Tom was so happy! They played with the ball together and became good friends.

### `Tom was sad because` — T=0.7 top-k=40
*{'words': 151, 'ended': True, 'distinct2': 0.86, 'rep4': 0.027}*

Tom was sad because he had no friends. He wanted to play with someone, but no one wanted to play with him. Tom walked around the park, looking for someone who would be his friend.

Then, Tom saw a little girl named Lily. She was playing on the swings and having fun. Tom went to her and said, "Hi, I am Tom. Can we play together?" Lily smiled and said, "Yes, let's play together!"

Tom and Lily played at the park all day. They laughed and had a great time. Tom was not lonely anymore. He had found a friend who liked to play with him, and they were both very happy. They played in the park every day, and they were never lonely again.

### `Tom was sad because` — T=0.8 top-p=0.95
*{'words': 91, 'ended': True, 'distinct2': 0.789, 'rep4': 0.057}*

Tom was sad because he could not find his coin. He looked everywhere, but it was gone. Just when he was about to give up, he saw a big dog with his coin in its mouth. The dog was running away with the coin in its mouth.

Tom tried to chase the dog, but he was too slow. The dog was too fast for Tom. He got tired and stopped running. Tom was sad because he lost his lucky coin.

### `Tom was sad because` — T=1.0 top-p=0.9
*{'words': 70, 'ended': True, 'distinct2': 0.841, 'rep4': 0.045}*

Tom was sad because he did not mean to make his friend Ben cry. He picked up the tube and put it away. He decided to be more careful and not spin it anymore. He learned that some things are not what they seem, and that some things are not meant for playing. He decided to spin the tube slower, and more slowly.

### `One day, a little bird` — greedy
*{'words': 151, 'ended': True, 'distinct2': 0.853, 'rep4': 0.014}*

One day, a little bird named Tim was flying in the sky. He saw a big tree with a lot of fruit. Tim wanted to eat the fruit, but he was too small to reach it. He saw a tall giraffe named Sam walking by.

"Sam, can you help me get the fruit?" asked Tim. Sam was happy to help. He used his long neck to reach the fruit and gave it to Tim. They both ate the fruit and it was very yummy.

As they ate, a big wind came and blew the fruit away. Tim and Sam were sad. But then, they saw a big pile of fruit on the ground. They were so happy and started to eat the fruit. The wind had brought them a surprise!

### `One day, a little bird` — T=0.7 top-k=40
*{'words': 133, 'ended': True, 'distinct2': 0.848, 'rep4': 0.008}*

One day, a little bird named Tim wanted to build a nest. He found a wide tree with many twigs. Tim asked his friend, Sam the squirrel, "Can I use the wide tree to build my nest?"

Sam said, "Yes, you can. But be careful. The tree is very tall."

Tim started to build his nest. He put a big twig on the ground. It was hard work. Tim was happy with his nest. But then, something unexpected happened.

A big wind came and blew the nest away. Tim was sad. He tried to build his nest, but it was not easy. The nest was gone forever.

### `One day, a little bird` — T=0.8 top-p=0.95
*{'words': 180, 'ended': True, 'distinct2': 0.698, 'rep4': 0.051}*

One day, a little bird was in a tree. The bird was sad. It wanted to fly like the other birds. The little bird said to a big bird, "I want to fly like you. Can you help me?"

The big bird said, "I will try. I will fly up and down and look for a way to help you." The little bird was happy. It tried to fly, but it was still very small.

Then, the big bird had an idea. It said, "I will help you fly high like me. But you must give me some food." The little bird was happy and gave the big bird some food. The big bird ate the food and flew up high. The little bird tried to fly like the big bird, and it worked! The little bird was not sad anymore. It had fun flying high with the big bird.

### `One day, a little bird` — T=1.0 top-p=0.9
*{'words': 124, 'ended': True, 'distinct2': 0.846, 'rep4': 0.008}*

One day, a little bird came to visit the tree. The bird was tired and needed a place to sleep. The tree said, "I recommend you go inside my leaves, and I will make you sleep."

The little bird was happy and said, "Thank you, tree!" The bird went inside the tree and took a long nap. When the bird woke up, it felt much better. The bird said, "I am not tired anymore. I can fly high in the sky again!"

The moral of the story is to listen to signs and not to do things that can make you tired.

### `The big red ball` — greedy
*{'words': 90, 'ended': True, 'distinct2': 0.843, 'rep4': 0.034}*

The big red ball said, "I am sorry, little blue ball. I did not mean to make you sad. Can we be friends?" The little blue ball said, "Yes, we can be friends. Let's play together!"

So, the big red ball and the little blue ball played together. They had fun and were happy. They learned that it is better to be friends than to fight. And they lived happily ever after.

### `The big red ball` — T=0.7 top-k=40
*{'words': 92, 'ended': True, 'distinct2': 0.747, 'rep4': 0.045}*

The big red ball wanted to help the small blue ball. They had a fun game to play. They would roll and bounce together. The small blue ball would bounce, and the big red ball would bounce. They were very happy playing together.

After a while, they got tired. The big red ball and the small blue ball sat down to rest. They were good friends. They knew they would play together again soon. And they lived happily ever after.

### `The big red ball` — T=0.8 top-p=0.95
*{'words': 81, 'ended': True, 'distinct2': 0.812, 'rep4': 0.051}*

The big red ball said, "I can help you tie your knot!" So, the big red ball tied the knot. The little blue ball was very happy and said, "Thank you, big red ball!"

Now, the big red ball and the little blue ball played together every day. They had so much fun and were always glad to be friends. And they lived happily ever after.

### `The big red ball` — T=1.0 top-p=0.9
*{'words': 41, 'ended': True, 'distinct2': 0.875, 'rep4': 0.026}*

The big red ball and the little blue car became best friends. They played together in the park every day. The big red ball was very happy to have a friend who loved to play in the park.

### `Mom said, "` — greedy
*{'words': 124, 'ended': True, 'distinct2': 0.862, 'rep4': 0.008}*

Mom said, "Yes, you can play with the ball, but be careful not to break it." Tim was very happy and started to play with the ball. He threw it up in the air and caught it. He was having so much fun.

Then, Tim saw a big tree with a lot of fruit. He wanted to eat the fruit, but he remembered what his mom said. Tim thought, "I should not eat the fruit. It is not good for me." So, he played with the ball instead. He had a great day at the park and learned to listen to his mom.

### `Mom said, "` — T=0.7 top-k=40
*{'words': 18, 'ended': True, 'distinct2': 1.0, 'rep4': 0.0}*

Mom said, "I love you too, Lily. You are my sweet girl."

### `Mom said, "` — T=0.8 top-p=0.95
*{'words': 239, 'ended': True, 'distinct2': 0.786, 'rep4': 0.03}*

Mom said, "Of course, Lily. Let's go and see the hotel."

They went to the hotel and met the nice lady at the door. Lily was so excited. She asked the lady, "Can I go in your room?" The nice lady said, "Of course, you can. But be careful, it's very dark and cold."

Lily went inside the hotel and saw many rooms. She saw a big room with lots of toys. Lily played with the toys and had so much fun. But then, she heard her mom calling her name. Lily ran back to her mom, but she couldn't find her mom. She was scared and cold.

Lily cried, "Mom, where are you?" The nice lady heard her and ran to the dark room. Lily's mom was there, looking for her. Lily hugged her mom and said, "I'm sorry, Mom. I should have listened to you."

From that day on, Lily learned to listen to her mom. She knew that her mom would be very worried and take her back to the big room in the hotel. They went home together, and Lily promised to always listen to her mom next time.

### `Mom said, "` — T=1.0 top-p=0.9
*{'words': 89, 'ended': True, 'distinct2': 0.83, 'rep4': 0.058}*

Mom said, "Yes, you can help me. You can put the real strawberries in the pot." Lily was so happy. She put the real strawberries in the pot with the real ones.

Lily's mom gave her some real strawberries to put in the pot. They cooked the strawberries together. They made the most delicious strawberries. Lily and her mom were very proud of their soup. They shared the strawberries with their friends and everyone loved them.

