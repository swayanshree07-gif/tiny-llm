# Sample suite — checkpoint step 40000

6 prompts x 4 settings, max 240 new tokens, seed 1234. `rep4` = fraction of repeated 4-grams (lower is better), `distinct2` = unique bigrams / bigrams, `ended` = model emitted `<|endoftext|>`.

## Summary by setting

| setting | mean words | ended | mean distinct2 | mean rep4 |
|---|---|---|---|---|
| greedy | 119 | 6/6 | 0.870 | 0.012 |
| T=0.7 top-k=40 | 130 | 6/6 | 0.820 | 0.049 |
| T=0.8 top-p=0.95 | 125 | 6/6 | 0.829 | 0.010 |
| T=1.0 top-p=0.9 | 151 | 5/6 | 0.833 | 0.020 |

## Samples

### `Once upon a time` — greedy
*{'words': 129, 'ended': True, 'distinct2': 0.93, 'rep4': 0.008}*

Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to play with. One day, she went to the park with her mom and took her ball with her.

At the park, Lily saw a boy named Tim. Tim was sad because he had no toys to play with. Lily wanted to help Tim, so she asked her mom if they could share her ball. Her mom said yes, and they played together.

After playing, Lily and Tim became good friends. They learned that sharing made them happy. The moral of the story is to always share with others and be kind.

### `Once upon a time` — T=0.7 top-k=40
*{'words': 163, 'ended': True, 'distinct2': 0.864, 'rep4': 0.019}*

Once upon a time, there was a little girl named Sally. Sally loved to play outside and look at the pretty flowers. One day, Sally saw a big tree with red and yellow fruits. She thought the fruits looked good to eat.

Sally wanted to pick the fruits, but she was too small. She tried to climb the tree, but she fell down. Sally felt sad and started to cry. Then, a friendly squirrel came by and asked, "Why are you sad, Sally?"

Sally told the squirrel, "I want to pick the fruits, but I can't reach them." The squirrel wanted to help Sally. He climbed the tree and picked a big, red apple for her. Sally was very happy and thanked the squirrel. They both sat under the tree and ate the yummy fruits together.

### `Once upon a time` — T=0.8 top-p=0.95
*{'words': 152, 'ended': True, 'distinct2': 0.907, 'rep4': 0.0}*

Once upon a time, there was a little girl named Amy. She loved to play in the park. One day, she saw a big slide. Amy was so happy! She ran to the slide and went up and down. She was having so much fun!

Amy saw a little boy named Tim at the park too. Tim was sad because he lost his toy. Amy wanted to help Tim feel happy. She thought of a plan. She would ask her mom if she could give the toy to Tim.

Amy's mom gave her some money. Amy went to the store and bought a new toy for Tim. Tim was very happy now! They played together in the park all day. The moral of the story is to always help others when they need it.

### `Once upon a time` — T=1.0 top-p=0.9
*{'words': 217, 'ended': True, 'distinct2': 0.792, 'rep4': 0.019}*

Once upon a time, there was a little girl named Mia. Mia had a pretty red collar. One day, she saw her friend Tom playing outside. Tom had a new blue collar. Mia felt envious. She wanted one too.

Mia went home and asked her mom to make her a pretty necklace for Tom. Her mom said yes. Mia was very happy. She did not know that her mom told her the necklace was magic. Mia put on the necklace and went back to play with Tom.

Mia and Tom played all day. But Mia's mom told her that the necklace was magic. She said it was not safe. Mia did not listen. When her mom was not looking, Mia took the necklace. She ran away and forgot about Tom.

Mia's mom found her and was sad. She told Mia that she should not have let Tom play with the necklace. Mia felt bad. She knew she was wrong. The magic necklace was not real. It was a bad thing. From that day, Mia learned to be kind and not to let Tom play with the things he wanted.

### `Lily and her dog went to the park.` — greedy
*{'words': 147, 'ended': True, 'distinct2': 0.808, 'rep4': 0.014}*

Lily and her dog went to the park. They saw a big tree with a hole in it. Lily said, "Let's look in the hole, Spot!" They went closer and looked inside. It was dark and a little scary.

Lily and Spot heard a noise. It was a tiny voice. The voice said, "Help me, please!" Lily and Spot looked at each other and smiled. They knew they had to help the voice.

Lily and Spot went into the hole. They found a small, scared bird. The bird said, "Thank you for helping me. I was stuck in the tree." Lily and Spot were happy they could help. They all became good friends and played together in the park.

### `Lily and her dog went to the park.` — T=0.7 top-k=40
*{'words': 244, 'ended': True, 'distinct2': 0.617, 'rep4': 0.112}*

Lily and her dog went to the park. They saw a big tree. Under the tree, there was a cat. The cat was sad. Lily asked, "Why are you sad, cat?" The cat said, "I can't find my way home."

Lily had an idea. "I can help you find your way home," she said. They walked together in the park. They saw a bird in a tree. The bird said, "I know the way. Follow me."

They followed the bird. They walked and walked. Then, something unexpected happened. The bird flew up to the top of the tree with Lily's. The cat was scared. But the bird said, "Don't worry, cat. I will help you find your home."

The bird and Lily looked and looked. They could not find the cat's home. The cat was sad. Lily had an idea. She said, "I will help you find your home, cat." The bird was happy. They looked and looked. And then, they saw the cat's home. The cat was happy. The cat said, "Thank you, Lily. You helped me find my home." They were both happy.

### `Lily and her dog went to the park.` — T=0.8 top-p=0.95
*{'words': 141, 'ended': True, 'distinct2': 0.821, 'rep4': 0.014}*

Lily and her dog went to the park. They played and ran around. Then, they saw a boy named Tim. Tim was sad. Lily asked, "Why are you sad, Tim?" Tim said, "I lost my toy."

Lily said, "Let's find your toy together." They looked and looked. Then, they found the toy behind a tree. Tim was happy. He said, "Thank you, Lily. You are a good friend."

Lily and Tim played together at the park. They had a fun day. When it was time to go home, Lily took Tim's hand. They walked back to their own houses. Lily and Tim were happy they played together.

### `Lily and her dog went to the park.` — T=1.0 top-p=0.9
*{'words': 158, 'ended': True, 'distinct2': 0.892, 'rep4': 0.006}*

Lily and her dog went to the park. They wanted to play catch. Lily threw the ball and the dog ran to catch it. They were very happy. But then, something unexpected happened. The ball hit a tree and broke open. Inside the ball, there was a big bubble!

The bubble grew and grew until it was very big. The big bubble popped, and out came a loud noise. Lily and her dog were scared. They ran away from the bubble. They ran back to their house to be safe.

When they got home, they saw that their house was full of balloons! Lily's mom had made balloons for them to play with. They laughed and played with the balloons all day. And they knew that sometimes, even when things seem bad, something good can happen.

### `Tom was sad because` — greedy
*{'words': 86, 'ended': True, 'distinct2': 0.847, 'rep4': 0.012}*

Tom was sad because he wanted to play with the ball. He tried to get the ball, but it was too high. He jumped and jumped, but he could not reach it. Tom was very sad.

Then, a big dog named Max came to help Tom. Max was strong and fast. He jumped and got the ball for Tom. Tom was so happy! They played with the ball together and had lots of fun.

### `Tom was sad because` — T=0.7 top-k=40
*{'words': 107, 'ended': True, 'distinct2': 0.925, 'rep4': 0.0}*

Tom was sad because he had no friends to play with. He looked around and saw a little girl named Lily. Lily was also lonely and looking for a friend. Tom thought for a moment and then decided to introduce himself.

Tom walked up to Lily and asked, "Do you want to be my friend?" Lily smiled and said, "Yes, I would love to be your friend!" They played together all day and had lots of fun. Tom was not lonely anymore, and he was very happy to have a friend.

### `Tom was sad because` — T=0.8 top-p=0.95
*{'words': 95, 'ended': True, 'distinct2': 0.84, 'rep4': 0.0}*

Tom was sad because he could not find his ball. He looked and looked, but the ball was still lost. Tom was sad and went back to his big house without his ball.

The next day, Tom went back to the park to look for his ball. But it was not there. He was very sad and started to cry. Tom's mom tried to make him feel better, but Tom was still sad. They never found the ball again, and Tom stayed sad.

### `Tom was sad because` — T=1.0 top-p=0.9
*{'words': 102, 'ended': True, 'distinct2': 0.901, 'rep4': 0.0}*

Tom was sad because he did not want to make his friend suffer. So, he decided to find his toy dog and bring it to Sam. Sam was happy to see Tom's toy dog and they both played with it.

They played all day and had lots of fun. At the end of the day, Tom and Sam were very tired. They went home and Tom's mom helped them find the toy dog. Tom was happy to have his toy dog back and not suffering anymore. They all had a good sleep.

### `One day, a little bird` — greedy
*{'words': 175, 'ended': True, 'distinct2': 0.845, 'rep4': 0.041}*

One day, a little bird named Tim was flying in the sky. He saw a big tree and decided to rest on a branch. While he was resting, he saw a big, red ball on the ground. Tim was very curious and wanted to play with the ball.

As Tim got closer to the ball, he saw a little girl named Sue. Sue was sad because she lost her ball. Tim wanted to help Sue, so he flew down and said, "Hi Sue, I found your ball. Let's play together!" Sue smiled and they started to play with the ball.

As they played, the ball suddenly popped! Tim and Sue were surprised. They looked at each other and laughed. Then, they found out that the ball was not a ball at all. It was a big, red balloon! They played with the balloon and had lots of fun.

### `One day, a little bird` — T=0.7 top-k=40
*{'words': 124, 'ended': True, 'distinct2': 0.797, 'rep4': 0.074}*

One day, a little bird named Tim wanted to find a new place to live. He flew around and saw a big tree. The tree had many branches and leaves. Tim thought it looked a good place to live.

Tim met a squirrel named Sam in the tree. Sam said, "This is a good place to live. Do you want to live here with me?" Tim said, "Yes, I want to live here with you."

Tim and Sam became good friends. They played and talked every day. Tim was very happy he found a new place to live. And they lived happily ever after.

### `One day, a little bird` — T=0.8 top-p=0.95
*{'words': 130, 'ended': True, 'distinct2': 0.752, 'rep4': 0.008}*

One day, a little bird was singing in the woods. The bird was happy and loved to sing. The sun was shining and the sky was blue. The bird felt so good that it wanted to sing more.

The bird started to sing a pretty song. A big bear heard the bird's song. The bear liked the song and wanted to be friends with the bird. But the bear did not know that the bird was scared of the bear.

The bird flew away to find another friend. But the bird was gone. The bird did not find a friend and never sang again. The bear was sad and walked away. The bird never sang again.

### `One day, a little bird` — T=1.0 top-p=0.9
*{'words': 129, 'ended': True, 'distinct2': 0.789, 'rep4': 0.032}*

One day, a little bird came to visit the garden. The bird was tired and needed a place to sleep. The wise sunflower listened to the bird and said, "I recommend you go to my home and rest."

The little bird flew up to the wise sunflower and said, "Thank you, wise sunflower. You are very kind." The sunflower smiled and said, "You are welcome, little bird. Come back and sleep in my favorite spot every day."

The little bird went back to her nest and slept in the wise sunflower's home. The sunflower was happy to help her friend. They both lived happily in the garden.

### `The big red ball` — greedy
*{'words': 52, 'ended': True, 'distinct2': 0.941, 'rep4': 0.0}*

The big red ball and the small blue ball were very happy. They played together all day long. They rolled and bounced and had lots of fun. At the end of the day, they were tired but happy. They knew they would play together again soon.

### `The big red ball` — T=0.7 top-k=40
*{'words': 125, 'ended': True, 'distinct2': 0.718, 'rep4': 0.09}*

The big red ball wanted to help the people and prevent them from getting into trouble. So, it asked the small blue ball to help. The small blue ball made a loud noise, and the big red ball roared. The big red ball was scared, but it wanted to help too.

The big red ball rolled very fast and hit the small blue ball. The small blue ball fell down, but it was not hurt. The people in the town saw what happened and they were very happy. They thanked the big red ball and played with it. The big red ball learned that helping others was a good thing to do.

### `The big red ball` — T=0.8 top-p=0.95
*{'words': 115, 'ended': True, 'distinct2': 0.842, 'rep4': 0.018}*

The big red ball came rolling near the yellow house. The ball hit the door and the door opened. Inside the house, there was a big room with many toys. The toys were all so happy to be out of the house.

But then, something unexpected happened. A little girl came out of the house with a toy that could talk. The girl was a magic girl who liked to make things disappear. The girl said, "I am here to make you happy, yellow house!" The yellow ball was so happy to be back with its friend, the girl.

### `The big red ball` — T=1.0 top-p=0.9
*{'words': 60, 'ended': True, 'distinct2': 0.898, 'rep4': 0.018}*

The big red ball rolled off the track. It was rolling down the hill very fast. Sue and Tim tried to catch the ball, but they could not. The big red ball hit a tree and broke into pieces. Tim and Sue were sad, and they could not play with the ball anymore.

### `Mom said, "` — greedy
*{'words': 126, 'ended': True, 'distinct2': 0.848, 'rep4': 0.0}*

Mom said, "Yes, you can. But be careful. Don't go too far."

Tim went to the park with his mom. He saw a big tree. He wanted to climb it. He started to climb, but he slipped. Tim fell down and hurt his knee. He cried, "Ouch! My knee!"

His mom came to help him. She said, "I told you to be careful, Tim." Tim's knee was hurt, but he was okay. They went home and Tim's mom put a band-aid on his knee. Tim felt better and said, "Thank you, Mom."

### `Mom said, "` — T=0.7 top-k=40
*{'words': 18, 'ended': True, 'distinct2': 1.0, 'rep4': 0.0}*

Mom said, "I love you too, Lily. You are my sweet girl."

### `Mom said, "` — T=0.8 top-p=0.95
*{'words': 116, 'ended': True, 'distinct2': 0.809, 'rep4': 0.018}*

Mom said, "Yes, but be careful with it." Tim played with the box, but he was not careful. He broke the box, and the tape was all over the floor. Tim was sad.

Tim's mom came and saw the mess. She said, "Tim, you need to be more careful with your toys. Now, the tape is all over the floor." Tim felt bad and helped his mom clean up the tape. From that day on, Tim was more careful with his toys, and he and his mom had fun playing with the tape together.

### `Mom said, "` — T=1.0 top-p=0.9
*{'words': 241, 'ended': False, 'distinct2': 0.725, 'rep4': 0.046}*

Mom said, "Yes, you can help me. But first, we need to go to the store and buy some gum. Do you want to come with me?"

Lily said, "Yes, I want to go with you. But I have a surprise for you. You have more gum than me. And you have me. You are my favorite."

Her mom said, "OK, let's go to the store. But you have to be fast. You have to use yourBeep and your speed. You have to ask your dad before you take them."

Lily said, "OK, mom. I will be fast. I want to use my bunny. And I want to buy gum with your own money."

They put on their coats and shoes and went to the store. They paid for their things and left the store. Lily was happy. She had her bunny and her new money. She was ready to go to her friend's birthday party. She had a lot of fun at the store. She saw many toys and books and games. She made a cake with her cute toys. She said, "Thank you, mom. This is the best birthday ever."

Her mom

