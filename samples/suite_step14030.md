# Sample suite — checkpoint step 14030

6 prompts x 4 settings, max 240 new tokens, seed 1234. `rep4` = fraction of repeated 4-grams (lower is better), `distinct2` = unique bigrams / bigrams, `ended` = model emitted `<|endoftext|>`.

## Summary by setting

| setting | mean words | ended | mean distinct2 | mean rep4 |
|---|---|---|---|---|
| greedy | 100 | 6/6 | 0.864 | 0.026 |
| T=0.7 top-k=40 | 133 | 5/6 | 0.753 | 0.082 |
| T=0.8 top-p=0.95 | 135 | 6/6 | 0.795 | 0.046 |
| T=1.0 top-p=0.9 | 125 | 6/6 | 0.842 | 0.024 |

## Samples

### `Once upon a time` — greedy
*{'words': 171, 'ended': True, 'distinct2': 0.765, 'rep4': 0.083}*

Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. One day, he found a big, red car in his toy box. He was very happy and wanted to play with it.

Tim took the big, red car outside to play. He put the car on the ground and pushed it around. The car went fast and made a loud noise. Tim laughed and clapped his hands. He played with the big, red car all day long.

As the sun went down, Tim's mom called him inside. Tim took the big, red car inside and put it on the table. He played with the car on the table and had lots of fun. When it was time to go home, Tim put the big, red car back in the toy box. He knew he would have more fun tomorrow.

### `Once upon a time` — T=0.7 top-k=40
*{'words': 126, 'ended': True, 'distinct2': 0.816, 'rep4': 0.016}*

Once upon a time, there was a little girl named Mia. Mia loved to play outside in the sun. One day, Mia saw a stranger. The stranger was not nice. Mia was scared.

Mia wanted to help the stranger. She thought if she could prevent the stranger from being mean, she would become brave. So, Mia went to find her mom. Her mom told her to stay close to the stranger.

Mia and her mom went to the stranger. Mia said sorry. The stranger smiled and said sorry. Mia learned that it is important to be brave and ask for help when you do not know about the stranger.

### `Once upon a time` — T=0.8 top-p=0.95
*{'words': 144, 'ended': True, 'distinct2': 0.79, 'rep4': 0.043}*

Once upon a time, there was a little girl named Amy. Amy loved to play in the park. One day, she saw a beautiful statue in the park. It was a statue of a nice man with a big hat.

Amy wanted to get closer to the statue, but she was too shy to ask her mom. She hid behind her mom and thought about the statue. Amy's mom saw her looking at the statue. She told Amy that it was a statue of a kind face.

Amy listened to her mom and went to the statue. She saw that the statue was smiling. Amy felt happy that she found the statue. She learned that it is good to ask for help when you are not sure about someone's.

### `Once upon a time` — T=1.0 top-p=0.9
*{'words': 140, 'ended': True, 'distinct2': 0.892, 'rep4': 0.007}*

Once upon a time, there was a little beetle named Tim. Tim had a shiny bell on his neck. Every day, Tim would look at the bell and admire it. He liked the bell a lot.

One day, Tim met a big beetle named Bob. Tim asked, "What is the bell on your neck, Bob?" Bob said, "The sun makes the bell glow. That is cool. I like to see the bracelet shine."

Tim and Bob became friends. They liked to look at the bell together. They played under the bright sun and felt safe. They had lots of fun with the sweet bell. And from that day on, they were the best of friends.

### `Lily and her dog went to the park.` — greedy
*{'words': 90, 'ended': True, 'distinct2': 0.899, 'rep4': 0.011}*

Lily and her dog went to the park. They saw a big tree with a hole in it. Lily said, "Look, dog! A hole! Let's go inside and see what's there." They went inside the hole and found a small, shiny box.

Lily and her dog opened the box. Inside, they found a shiny ball. They were very happy and played with the ball all day. The park was a fun place for them.

### `Lily and her dog went to the park.` — T=0.7 top-k=40
*{'words': 156, 'ended': True, 'distinct2': 0.677, 'rep4': 0.059}*

Lily and her dog went to the park. They saw a big tree with a swing. Lily liked the swing. She said, "I want to play on the swing, please." Her mom said, "You can play on the swing, but be careful."

Lily went on the swing and saw a boy with a big blue hat. The boy said, "I like the blue hat, but you are too small." Lily said, "I want to show you how to play on the swing." The boy smiled and said, "Okay, show me."

Lily helped the boy get on the swing. The boy was happy. He said, "Thank you!" They played on the swing together. They had a fun day at the park.

### `Lily and her dog went to the park.` — T=0.8 top-p=0.95
*{'words': 174, 'ended': True, 'distinct2': 0.821, 'rep4': 0.023}*

Lily and her dog went to the park. They saw a big, sour lemon on the ground. Lily wanted to taste it, so she picked it up. She took a bite, and it made her face funny. Lily did not like sour things.

Lily's mom came to help her. She said, "Lily, do not eat the lemon. It is not for you." Lily nodded and said, "Okay, mommy. I want to try it."

Lily took a small bite of the sour lemon. But it was not sweet like her hand. It was sour and made her mouth feel funny. She said, "Ew, it is sour!" Lily's mom laughed and said, "That's because you don't like sour things. Let's go home and cook something yummy." Lily and her mom left the park and went home, feeling happy and safe.

### `Lily and her dog went to the park.` — T=1.0 top-p=0.9
*{'words': 214, 'ended': True, 'distinct2': 0.784, 'rep4': 0.043}*

Lily and her dog went to the park. They saw a big slide and a swing. Lily was very happy. She climbed up the ladder and sat on the slide. She pushed with her nose and sat on the slide. She felt very brave and brave.

"Ready, set, go!" she said. slide down fast and felt the wind in her hair. She laughed and ran back to the ladder. She saw the other kids and felt a little scared. She looked around and saw the sun and the sky. She wished she could play on the slide.

"Hello, do you want to play with me?" she asked the other kids. She was very friendly and kind.

The other kids were surprised and surprised. They nodded and smiled. They took turns on the slide. They pretended they were slides down fast and everyone had a lot of fun.

Lily thanked them for joining her. She climbed up and slid down the slide. She felt like a brave and independent kid. She learned how to slide faster and farther. She was happy and proud.

### `Tom was sad because` — greedy
*{'words': 110, 'ended': True, 'distinct2': 0.936, 'rep4': 0.0}*

Tom was sad because he lost his toy. He looked everywhere for it, but he could not find it. He asked his friend, a dog named Max, to help him. Max was a good friend and wanted to help Tom.

Max and Tom looked for the toy together. They looked under the bed, behind the couch, and in the toy box. Finally, they found the toy in a big tree. Tom was so happy! He said, "Thank you, Max!" They played together with the toy and had a fun day.

### `Tom was sad because` — T=0.7 top-k=40
*{'words': 98, 'ended': True, 'distinct2': 0.897, 'rep4': 0.021}*

Tom was sad because he had no friends to play with. He looked around and saw a little girl named Lily. Lily was also lonely and needed a friend. Tom and Lily walked over to Lily and said, "Hi! Do you want to play with me?"

Lily smiled and said, "Yes, I want to play too!" They played together and had lots of fun. Tom and Lily became good friends. They learned that being kind and not ugly can make you happy.

### `Tom was sad because` — T=0.8 top-p=0.95
*{'words': 107, 'ended': True, 'distinct2': 0.84, 'rep4': 0.038}*

Tom was sad because he could not find his coin. He looked everywhere, but he could not find it. He felt sad and said, "I lost my coin and now I have nothing."

Just then, a nice dog came running to Tom. The dog had a shiny coin in its mouth! Tom was so happy and said, "Thank you, dog!" The dog wagged its tail and gave Tom his coin. Tom smiled and said, "You're welcome, dog!" They became friends and played together all day.

### `Tom was sad because` — T=1.0 top-p=0.9
*{'words': 63, 'ended': True, 'distinct2': 0.935, 'rep4': 0.0}*

Tom was sad because he did not want to make his friend sad. He wanted to play with her. So, Tom went to his room and got a new toy car. He played with the toy car all day and did not miss it at night. Tom learned that being kind is more important than being kind.

### `One day, a little bird` — greedy
*{'words': 125, 'ended': True, 'distinct2': 0.871, 'rep4': 0.0}*

One day, a little bird named Tim was flying in the sky. He saw a big tree with a small door on it. Tim was curious and wanted to see what was inside. He flew down and opened the door. Inside, he found a room full of yummy food.

Tim started to eat the food, but he heard a loud noise. It was a big, scary bear! The bear was hungry too. Tim was scared, but he had an idea. He flew away and found a big, yummy cake. Tim ate the cake and felt happy. He knew that he could always find food in the dark.

### `One day, a little bird` — T=0.7 top-k=40
*{'words': 89, 'ended': True, 'distinct2': 0.886, 'rep4': 0.023}*

One day, a little bird named Tim wanted to build a nest for his eggs. He needed a cozy branch to make the nest. Tim asked his friend, Cat, to help him find a new branch for his nest.

They looked and looked, and finally, they found a new branch for the nest. Tim and Cat were very happy. The nest was warm and safe in the tree. They played together all day and became best friends.

### `One day, a little bird` — T=0.8 top-p=0.95
*{'words': 163, 'ended': True, 'distinct2': 0.735, 'rep4': 0.087}*

One day, a little bird was singing a happy song. A man in the hat came to sit on a tree. The bird looked scared, but it started to sing too. The man smiled and said, "I like your song."

The bird sang even louder. The man did not think he was funny. He did not want to make the bird sad. So, the man took the bird home. He gave the bird a soft bed and a warm blanket.

The next day, the man took the bird to a new tree. The bird was surprised to see the man. The bird said, "Thank you for being nice to me." The man smiled and said, "You're welcome." The bird sang even louder.

The moral of the story is to always be kind and help others.

### `One day, a little bird` — T=1.0 top-p=0.9
*{'words': 199, 'ended': True, 'distinct2': 0.778, 'rep4': 0.01}*

One day, a little bird came to visit the barn. The bird was tired and needed a rest. The bird saw a big rock near the hay and thought, "I will rest on this rock and sleep here."

The bird sat on the rock and closed its eyes. While the bird was resting, a big wind came. The wind blew the hay away. The bird woke up and saw that the hay was gone. The bird was sad.

The bird flew down to the barn to find the hay. When the bird found the hay, something unexpected happened. The hay started to grow bigger and bigger! The bird was surprised. The hay was not hay at all. It was a big, sleepy cat who was sleeping on the hay all along. The cat woke up and said, "You found my hay, but you made it so much bigger. Now, I can sleep on the hay." The bird and the cat became best friends, and they lived happily ever after.

### `The big red ball` — greedy
*{'words': 46, 'ended': True, 'distinct2': 0.867, 'rep4': 0.047}*

The big red ball was very happy to be found. It rolled and rolled until it found a new place to play. The big red ball was not alone anymore. It had a new friend, and they played together every day.

### `The big red ball` — T=0.7 top-k=40
*{'words': 242, 'ended': False, 'distinct2': 0.39, 'rep4': 0.351}*

The big red ball wanted to play with the little blue ball. But the little blue ball did not want to play with the big red ball. The big red ball was sad. The little blue ball wanted to play with the big red ball, but the big red ball did not want to. The little blue ball wanted to play with the big red ball.

The little blue ball had a plan. It would pass the big red ball to the big red ball. The little blue ball knew the big red ball was not nice. The little blue ball rolled to the big red ball. The big red ball saw the little blue ball and started to play with it. The big red ball saw the little blue ball playing and wanted to play too. The big red ball said, "No, you can't play with me. You are too small." The little blue ball was sad but went away.

The big red ball kept playing and passing the small blue ball. But soon, the big red ball got too close to the little blue ball. The little blue ball got too close to the big red ball. The big red ball was hit by the small blue ball. The little blue ball was hurt and could

### `The big red ball` — T=0.8 top-p=0.95
*{'words': 53, 'ended': True, 'distinct2': 0.769, 'rep4': 0.04}*

The big red ball came rolling towards the yellow dog. The dog got scared and ran away. The yellow dog was safe. The yellow dog and the big red ball became friends. They played together every day. And the yellow dog always had fun in the garden.

### `The big red ball` — T=1.0 top-p=0.9
*{'words': 87, 'ended': True, 'distinct2': 0.791, 'rep4': 0.06}*

The big red ball rolled off, and Max chased after it. The big red ball rolled back and forth, faster and faster. Max was so happy to catch the big red ball.

Max played with the big red ball for the whole day. He showed all the other cars his new toy, and they all thought he looked like a tiny cat too. Max was proud of himself for catching the big red ball and playing with it.

### `Mom said, "` — greedy
*{'words': 60, 'ended': True, 'distinct2': 0.847, 'rep4': 0.018}*

Mom said, "I'm sorry, Tim. I didn't mean to make you sad." Tim's mom said, "It's okay, Tim. We all make mistakes. Let's clean up the mess together." They cleaned up the mess and Tim's mom gave him a new toy. Tim was happy again.

### `Mom said, "` — T=0.7 top-k=40
*{'words': 88, 'ended': True, 'distinct2': 0.851, 'rep4': 0.024}*

Mom said, "I think the toy is hiding. Let's look for it together!" So, Tim and his mom went outside to look for the toy. They looked under the trees and behind the bushes.

After a while, they found the toy. Tim was very happy. He said, "Thank you, Mom!" Tim and his mom went home and played with the toy together. They had a fun day in the sun.

### `Mom said, "` — T=0.8 top-p=0.95
*{'words': 169, 'ended': True, 'distinct2': 0.815, 'rep4': 0.042}*

Mom said, "We are making a cake with milk and sugar. But we need to put on some hats and gloves. Mom says to be careful with the thermometer. The thermometer says it is hot. The thermometer says it is too hot for you."

Lily and Ben nod. They help Mom cut the cake into shapes. They put the ice cream in one half and some on top. They look at the shapes. They see the sun, the shapes, and the shapes. They are happy.

They go to the kitchen to get a drink. They drink some water. They wash their hands. They feel better. They say, "Thank you, Mom. You are the best."

Mom smiles and says, "You are welcome, Lily. You are the best too. Now let's go to the kitchen and eat some."

### `Mom said, "` — T=1.0 top-p=0.9
*{'words': 49, 'ended': True, 'distinct2': 0.875, 'rep4': 0.022}*

Mom said, "Tom, you should not mark your shoes. Now you need to put them away." Tom felt sad. His mom said, "Next time, let me show you something better." Tom learned to be more careful.

