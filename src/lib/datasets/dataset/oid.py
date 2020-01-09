from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class OID(data.Dataset):
  num_classes = 545
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(COCO, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'oid')
    self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 
      '{}-annotations-bbox.json.json').format(split)
    self.max_objs = 128
    self.class_name = ['__background__', 'Tortoise', 'Magpie', 'Sea turtle',
      'Football', 'Ambulance', 'Ladder', 'Toothbrush', 'Sink', 'Toy',
      'Organ (Musical Instrument)', 'Apple', 'Human eye', 'Cosmetics',
      'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Human beard', 'Bird',
      'Parking meter', 'Traffic light', 'Croissant', 'Cucumber', 'Radish',
      'Towel', 'Doll', 'Skull', 'Washing machine', 'Glove', 'Tick', 'Belt',
      'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle',
      'Home appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot', 'Headphones',
      'Hot dog', 'Shorts', 'Fast food', 'Bus', 'Boy', 'Bicycle wheel', 'Barge',
      'Laptop', 'Miniskirt', 'Drill (Tool)', 'Dress', 'Bear', 'Waffle',
      'Pancake', 'Brown bear', 'Woodpecker', 'Blue jay', 'Pretzel', 'Bagel',
      'Tower', 'Teapot', 'Person', 'Bow and arrow', 'Swimwear', 'Beehive',
      'Brassiere', 'Bee', 'Bat (Animal)', 'Starfish', 'Popcorn', 'Burrito',
      'Chainsaw', 'Balloon', 'Tent', 'Vehicle registration plate', 'Lantern',
      'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 'Scissors',
      'Stairs', 'Computer keyboard', 'Printer', 'Traffic sign', 'Chair',
      'Shirt', 'Poster', 'Cheese', 'Sock', 'Fire hydrant', 'Land vehicle',
      'Earrings', 'Tie', 'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin',
      'Bidet', 'Snack', 'Snowmobile', 'Clock', 'Medical equipment', 'Cattle',
      'Cello', 'Jet ski', 'Camel', 'Coat', 'Suit', 'Desk', 'Cat',
      'Bronze sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon',
      'Computer mouse', 'Cookie', 'Office building', 'Fountain', 'Coin',
      'Calculator', 'Cocktail', 'Computer monitor', 'Box', 'Christmas tree',
      'Cowboy hat', 'Hiking equipment', 'Studio couch', 'Drum', 'Dessert',
      'Wine rack', 'Drink', 'Zucchini', 'Human mouth', 'Dairy Product', 'Dice',
      'Oven', 'Dinosaur', 'Couch', 'Whiteboard', 'Door', 'Hat', 'Shower',
      'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero',
      'Tin can', 'Mug', 'Tap', 'Harbor seal', 'Stretcher', 'Goggles',
      'Human body', 'Roller skates', 'Coffee cup', 'Cutting board', 'Blender',
      'Plumbing fixture', 'Stop sign', 'Office supplies', 'Volleyball (Ball)',
      'Vase', 'Slow cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Paper towel',
      'Personal care', 'Food', 'Sun hat', 'Tree house', 'Flying disc', 'Skirt',
      'Gas stove', 'Salt and pepper shakers', 'Mechanical fan', 'Fruit',
      'French fries', 'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill',
      'Fox', 'Flag', 'French horn', 'Window blind', 'Human foot', 'Golf cart',
      'Jacket', 'Egg (Food)', 'Street light', 'Guitar', 'Pillow', 'Human leg',
      'Grape', 'Human ear', 'Power plugs and sockets', 'Panda', 'Giraffe',
      'Woman', 'Door handle', 'Rhinoceros', 'Bathtub', 'Goldfish',
      'Houseplant', 'Goat', 'Baseball bat', 'Baseball glove', 'Mixing bowl',
      'Marine invertebrates', 'Kitchen utensil', 'House', 'Horse',
      'Stationary bicycle', 'Hammer', 'Ceiling fan', 'Sofa bed',
      'Adhesive tape', 'Harp', 'Sandal', 'Bicycle helmet', 'Saucer',
      'Harpsichord', 'Human hair', 'Hamster', 'Curtain', 'Bed', 'Kettle',
      'Fireplace', 'Scale', 'Drinking straw', 'Insect', 'Kitchenware',
      'Invertebrate', 'Food processor', 'Bookcase', 'Refrigerator',
      'Wood-burning stove', 'Punching bag', 'Common fig', 'Jaguar (Animal)',
      'Golf ball', 'Fashion accessory', 'Alarm clock', 'Filing cabinet',
      'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife',
      'Bottle', 'Lynx', 'Lavender (Plant)', 'Lighthouse', 'Dumbbell',
      'Human head', 'Bowl', 'Porch', 'Lizard', 'Billiard table', 'Mammal',
      'Mouse', 'Motorcycle', 'Musical instrument', 'Swim cap', 'Frying pan',
      'Snowplow', 'Bathroom cabinet', 'Missile', 'Bust', 'Man', 'Milk',
      'Plate', 'Mobile phone', 'Baked goods', 'Mushroom',
      'Pitcher (Container)', 'Mirror', 'Personal flotation device',
      'Table tennis racket', 'Pencil case', 'Musical keyboard', 'Scoreboard',
      'Briefcase', 'Kitchen knife', 'Nail (Construction)', 'Tennis ball',
      'Plastic bag', 'Oboe', 'Chest of drawers', 'Ostrich', 'Piano', 'Girl',
      'Plant', 'Potato', 'Sports equipment', 'Pasta', 'Penguin', 'Pumpkin',
      'Pear', 'Infant bed', 'Polar bear', 'Mixer', 'Cupboard', 'Jacuzzi',
      'Pizza', 'Digital clock', 'Pig', 'Reptile', 'Rifle', 'Lipstick',
      'Skateboard', 'Raven', 'High heels', 'Red panda', 'Rose', 'Rabbit',
      'Sculpture', 'Saxophone', 'Shotgun', 'Seafood', 'Submarine sandwich',
      'Snowboard', 'Sword', 'Picture frame', 'Sushi', 'Loveseat', 'Ski',
      'Squirrel', 'Tripod', 'Scorpion', 'Segway', 'Training bench', 'Snake',
      'Coffee table', 'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea',
      'Tank', 'Taco', 'Telephone', 'Tiger', 'Strawberry', 'Trumpet', 'Tree',
      'Tomato', 'Train', 'Tool', 'Picnic basket', 'Trousers',
      'Bowling equipment', 'Football helmet', 'Truck', 'Coffeemaker', 'Violin',
      'Vehicle', 'Handbag', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale',
      'Zebra', 'Auto part', 'Jug', 'Monkey', 'Lion', 'Bread', 'Platter',
      'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle',
      'Hippopotamus', 'Crocodile', 'Toilet', 'Toilet paper', 'Clothing',
      'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket',
      'Wine glass', 'Countertop', 'Tablet computer', 'Waste container',
      'Swimming pool', 'Dog', 'Book', 'Elephant', 'Shark', 'Candle', 'Leopard',
      'Axe', 'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm tree',
      'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 'Garden Asparagus',
      'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster',
      'Convenience store', 'Bench', 'Ice cream', 'Caterpillar', 'Butterfly',
      'Parachute', 'Orange', 'Antelope', 'Moths and butterflies', 'Window',
      'Closet', 'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach',
      'Coconut', 'Seat belt', 'Raccoon', 'Fork', 'Lamp', 'Camera',
      'Squash (Plant)', 'Racket', 'Human face', 'Human arm', 'Vegetable',
      'Diaper', 'Unicycle', 'Falcon', 'Snail', 'Shellfish', 'Cabbage',
      'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool',
      'Envelope', 'Cake', 'Dragonfly', 'Common sunflower', 'Microwave oven',
      'Honeycomb', 'Marine mammal', 'Sea lion', 'Ladybug', 'Shelf', 'Watch',
      'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 'Van', 'Light bulb',
      'Corded phone', 'Sports uniform', 'Tennis racket', 'Wall clock',
      'Serving tray', 'Kitchen & dining room table', 'Dog bed', 'Cake stand',
      'Cat furniture', 'Bathroom accessory', 'Kitchen appliance', 'Tire',
      'Ruler', 'Luggage and bags', 'Microphone', 'Broccoli', 'Umbrella',
      'Pastry', 'Grapefruit', 'Animal', 'Bell pepper', 'Turkey', 'Lily',
      'Pomegranate', 'Doughnut', 'Glasses', 'Human nose', 'Pen', 'Ant', 'Car',
      'Aircraft', 'Human hand', 'Teddy bear', 'Watermelon', 'Cantaloupe',
      'Flute', 'Balance beam', 'Sandwich', 'Shrimp', 'Sewing machine',
      'Binoculars', 'Rays and skates', 'Ipod', 'Accordion', 'Willow', 'Crab',
      'Crown', 'Seahorse', 'Perfume', 'Alpaca', 'Taxi', 'Canoe',
      'Remote control', 'Wheelchair', 'Rugby ball', 'Helmet',
    ]
    self._valid_ids = [
      1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
      42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
      79, 80, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
      114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
      128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142,
      143, 144, 145, 146, 148, 149, 150, 151, 152, 154, 158, 160, 161, 162,
      164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 177, 178,
      179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
      193, 194, 195, 196, 197, 198, 199, 200, 203, 204, 205, 206, 207, 208,
      209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223,
      224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
      238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252,
      253, 256, 257, 258, 259, 260, 261, 262, 263, 265, 267, 268, 269, 270,
      271, 272, 273, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
      286, 288, 289, 290, 291, 292, 293, 295, 296, 297, 298, 299, 300, 301,
      302, 303, 304, 305, 306, 307, 308, 310, 312, 313, 314, 315, 317, 318,
      319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
      333, 334, 335, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347,
      348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
      362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377,
      378, 379, 380, 381, 382, 383, 384, 385, 386, 388, 389, 390, 391, 392,
      393, 394, 395, 397, 398, 399, 400, 402, 403, 404, 405, 407, 408, 409,
      410, 411, 412, 413, 414, 415, 418, 419, 420, 421, 422, 423, 424, 425,
      426, 427, 428, 429, 430, 431, 433, 434, 435, 436, 437, 438, 439, 440,
      441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 455, 456,
      457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470,
      471, 472, 474, 476, 477, 478, 479, 480, 481, 482, 484, 485, 486, 487,
      488, 489, 490, 491, 492, 493, 494, 495, 497, 498, 499, 500, 501, 502,
      503, 504, 505, 506, 507, 509, 510, 511, 512, 513, 514, 515, 516, 517,
      518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531,
      532, 533, 534, 535, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,
      548, 551, 552, 553, 554, 555, 556, 557, 558, 559, 561, 562, 563, 564,
      565, 566, 567, 568, 569, 570, 571, 572, 573, 575, 576, 577, 579, 580,
      581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594,
      595, 596, 597, 598, 601,
    ]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 81 * 36, (v // 9) % 9 * 31, v % 9 * 31) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing open images dataset v5 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
