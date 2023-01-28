import cv2
import os
import jax
import jax.numpy as jnp
import random
#import cPickle as pickle
import pickle
import warnings
import argparse


# Set up arguments for the dataset generator to parse through.
parser = argparse.ArgumentParser(description='Sort-of-CLEVR dataset generator') # Adds semantic description for the batch of arguments
parser.add_argument('--seed', type=int, default=1, metavar='S', # Adds a seed argument that will be used for all random generation instances
                    help='random seed (default: 1)')
parser.add_argument('--t-subtype', type=int, default=-1, # Add the argument to represent ternary questions as an int.
                    help='Force ternary questions to be of a given type')
args = parser.parse_args() # Parse the argument batch into a callable object

random.seed(args.seed) # Initialize random seed for generator
rand_key = jax.random.PRNGKey(args.seed) # Same as above but set for jax

# Metadata for the scope of our experiment
train_size = 9800 # Training dataset size
test_size = 200 # Testing dataset size
img_size = 75 # Image size (pixels) (75x75)
size = 5 # Size of object in an image (radius)
question_size = 18  ## 2 x (6 for one-hot vector of color), 3 for question type, 3 for question subtype
q_type_idx = 12 # Index of the question type (non-relational[12] vs. relational[13] vs. ternary relational[14]) 
sub_q_type_idx = 15  # Index of the question subtype (last 3 bits of encoded vector) 
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""


nb_questions = 10 # Number of non-relational questions
dirs = './data' # Directory for saving dataset

# Vector of possible colors (6) for our use case  
colors = [
    (0,0,255),##r; Red
    (0,255,0),##g; Green
    (255,0,0),##b; Blue
    (0,156,255),##o; Orange 
    (128,128,128),##k; Gray
    (0,255,255)##y; Yellow
]

# Create directory for data 
try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

# Function to create centroids of each object in an image
def center_generate(objects):
    while True:
        pas = True # Flag variable to control loop
        center = jax.random.randint(rand_key, (2, ), 0+size, img_size - size) # Generate a random (x,y) centroid, we want some padding for each object, so the minimum of an x or y point will be equivalent to the variable size (5).       
        if len(objects) > 0: # In the case where the passed in objects argument is not empty:
            for name,c,shape in objects: # Loop through the object set
                # If the vector sum of the squared difference between the object's center and the randomly generated center is less than 100 ((5*2)^2), set flag variable to false and continue the while loop
                if ((center - c) ** 2).sum() < ((size * 2) ** 2): # This ensures that there is an appropriate amount of space between the new centroid and the existing objects
                    pas = False
        if pas:
            return center # Return new centroid to apply to the next object


# Dataset generator
def build_dataset():
    '''State Description Matrix - We will run the model using the objects' states (color, centroid, shape, size) persisted in an array instead of the 2D pixel image format'''
    STATE = {}
    objects = [] # Initialize empty array of objects
    img = jnp.ones((img_size,img_size,3)) * 255 # Initialize matrix representing image's pixels with shape (75x75x3), where 3 represents the RGB encoding. We multiply by 255 to create a white canvas (White=(255,255,255)).
    # Loop through vector of eligible colors
    for color_id,color in enumerate(colors):  
        center = center_generate(objects) # Initialize a centroid for an object for each color in the loop
        idx = len(STATE.keys())
        if random.random()<0.5: # If random value is less than 0.5, we will add a rectangle to the image
            start = (center[0]-size, center[1]-size) # Offset the start of the rectangle by the size variable value.
            end = (center[0]+size, center[1]+size) # Offset the end of rectangle using size
            cv2.rectangle(img, start, end, color, -1) # Insert rectangle using dimensions above. 
            objects.append((color_id,center,'r')) # Add the new rectangle to the objects array, denoted by its color, center, and 'r' to represent rectangle
            STATE[idx] = {'id': color_id, 'center': center, 'shape': 'r', 'size': size}
        else: # If random value is greater than or equal to 0.5, we will add a circle to the image instead
            center_ = (center[0], center[1]) # Deconstruct generated centroid
            cv2.circle(img, center_, size, color, -1) # Add circle to the image
            objects.append((color_id,center,'c')) # Add the new circle to the objects array, with 'c' to represent circle. 
            STATE[idx] = {'id': color_id, 'center': center, 'shape': 'c', 'size': size}
    # Initialize empty vectors to store questions and answers
    ternary_questions = []
    binary_questions = []
    norel_questions = []
    ternary_answers = []
    binary_answers = []
    norel_answers = []
    """Non-relational questions"""
    for _ in range(nb_questions):
        question = jnp.zeros((question_size)) # Initialize a vector of zeroes (size: 18) which will be used to encode our questions in binary form
        color = random.randint(0,5) # Choose a random color with 6 possible choices; The first 6 bits of the question vector will represent a one-hot vector for the color
        question[color] = 1 # Set the corresponding index of the color to 1
        question[q_type_idx] = 1 # The 12th index will be set to 1 to indicate that the question is non-relational
        '''
        Possible subtypes for nonrelational:
        1) Shape of certain colored object
        2) Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image
        3) Vertical location of certain colored object : whether it is on the upside of the image or downside of the image
        '''
        subtype = random.randint(0,2) # Choose from the 3 subtypes listed above.
        question[subtype+sub_q_type_idx] = 1 # Set subtype index (last 3 bits of encoding)
        norel_questions.append(question) # Add the question to the vector
        """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
        if subtype == 0:
            """query shape->rectangle/circle"""
            if STATE[color]['shape'] == 'r':
                answer = 2 # Represents that the object is a rectangle
            else:
                answer = 3 # Represents that the object is a circle

        elif subtype == 1: 
            """query horizontal position->yes/no"""
            if STATE[color]['center'][0] < img_size / 2: # Check to see if the x coordinate of object's centroid falls closer to left side or right side of image
                answer = 0 # Answer: Left
            else:
                answer = 1 # Answer: Right

        elif subtype == 2:
            """query vertical position->yes/no"""
            if STATE[color]['center'][1] < img_size / 2: # Check to see if the y coordinate of object's centroid falls closer to top side or bottom side of image
                answer = 0 # Answer: Top
            else:
                answer = 1 # Answer: Bottom
        norel_answers.append(answer) # Add nonrelational answer to vector 
    
    """Binary Relational questions; NOTE: Refer to code for nonrelational question setup above, as some code is the same"""
    for _ in range(nb_questions):
        question = jnp.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx+1] = 1 # Set the 13th index to 1 to represent binary relational type
        '''
        Possible subtypes for relational:
        1) Shape of the object which is closest to the certain colored object
        2) Shape of the object which is furthest to the certain colored object
        3) Number of objects which have the same shape with the certain colored object
        '''
        subtype = random.randint(0,2) 
        question[subtype+sub_q_type_idx] = 1
        binary_questions.append(question)
        """Answer : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]; NOTE: There are no yes/no answers for binary relational but we will include so that answer vector length remains as 10"""
        if subtype == 0:
            """closest-to->rectangle/circle"""
            my_obj = STATE[color]['center'] # Extract the centroid of a certain object which is chosen based on color index
            dist_list = [((my_obj - STATE[obj]['center']) ** 2).sum() for obj in STATE] # Store the distance between my_obj and every other object in the set in dist_list
            dist_list[dist_list.index(0)] = 999 # The element with distance 0 will be the same object as the comparable object, so set to 999 and disregard
            closest = dist_list.index(min(dist_list)) # Extract the smallest distance
            if STATE[closest]['shape'] == 'r': # Check to see if closest object is a rectangle
                answer = 2 
            else: # Check to see if closest object is a circle
                answer = 3
                
        elif subtype == 1:
            """furthest-from->rectangle/circle"""
            my_obj = STATE[color]['center']
            dist_list = [((my_obj - STATE[obj]['center']) ** 2).sum() for obj in STATE] # Store the distance between my_obj and every other object in the set in dist_list
            furthest = dist_list.index(max(dist_list)) # Extract the largest distance
            if STATE[furthest]['shape'] == 'r': # Check to see if farthest object is a rectangle
                answer = 2
            else: # Check to see if farthest object is a circle
                answer = 3

        elif subtype == 2:
            """count->1~6"""
            my_obj = STATE[color]['shape'] # Store object to compare the rest to chosen object
            count = -1 # Set to negative one to account for the identical object in the set under the for loop
            for obj in STATE:
                if STATE[obj]['shape'] == my_obj: # Loop through each object in the state table and check if shape is the same
                    count +=1 # Increment # of same objects
            answer = count+4 # Add 4 to the count to adjust to the appropriate index value of the answer vector

        binary_answers.append(answer)

    """Ternary Relational questions; NOTE: Refer to code for nonrelational question setup above, as some code is the same"""
    for _ in range(nb_questions):
        question = jnp.zeros((question_size))
        rnd_colors = jax.random.permutation(rand_key, jnp.arange(5)) # Randomly permutate color scheme to extract color designations for objects
        # 1st object color assignment
        color1 = rnd_colors[0]
        question[color1] = 1 # Assign color for first one-hot vector
        # 2nd object color assignment
        color2 = rnd_colors[1]
        question[6 + color2] = 1 # Assign color for second one-hot vector

        question[q_type_idx + 2] = 1 # Set 14th index to 1 to denote ternary relational question type
        
        # Check to see if ternary question subtype was passed into argument parser
        if args.t_subtype >= 0 and args.t_subtype < 3: 
            subtype = args.t_subtype # Set subtype index
        else: 
            subtype = random.randint(0, 2) # Randomly generate subtype  

        question[subtype+sub_q_type_idx] = 1 
        ternary_questions.append(question)

        # get coordiantes of object from question
        A = STATE[color1][1]
        B = STATE[color2][1]
        
        """Answer : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]"""
        if subtype == 0:
            """between->1~4"""

            between_count = 0 
            # check is any objects lies inside the box
            for other_obj in objects:
                # skip object A and B
                if (STATE[other_obj][0] == color1) or (STATE[other_obj][0] == color2):
                    continue

                # Get x and y coordinate of third object
                other_objx = STATE[other_obj][1][0]
                other_objy = STATE[other_obj][1][1]

                if (A[0] <= other_objx <= B[0] and A[1] <= other_objy <= B[1]) or \
                   (A[0] <= other_objx <= B[0] and B[1] <= other_objy <= A[1]) or \
                   (B[0] <= other_objx <= A[0] and B[1] <= other_objy <= A[1]) or \
                   (B[0] <= other_objx <= A[0] and A[1] <= other_objy <= B[1]):
                    between_count += 1

            answer = between_count + 4
        elif subtype == 1:
            """is-on-band->yes/no"""
            
            grace_threshold = 12  # half of the size of objects
            epsilon = 1e-10  
            m = (B[1]-A[1])/((B[0]-A[0]) + epsilon ) # add epsilon to prevent dividing by zero
            c = A[1] - (m*A[0])

            answer = 1  # default answer is 'no'

            # check if any object lies on/close the line between object A and object B
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                other_obj_pos = other_obj[1]
                
                # y = mx + c
                y = (m*other_obj_pos[0]) + c
                if (y - grace_threshold)  <= other_obj_pos[1] <= (y + grace_threshold):
                    answer = 0
        elif subtype == 2:
            """count-obtuse-triangles->1~6"""

            obtuse_count = 0

            # disable warnings
            # the angle computation may fail if the points are on a line
            warnings.filterwarnings("ignore")
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                # get position of 3rd object
                C = other_obj[1]
                # edge length
                a = jnp.linalg.norm(B - C)
                b = jnp.linalg.norm(C - A)
                c = jnp.linalg.norm(A - B)
                # angles by law of cosine
                alpha = jnp.rad2deg(jnp.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
                beta = jnp.rad2deg(jnp.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
                gamma = jnp.rad2deg(jnp.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
                max_angle = max(alpha, beta, gamma)
                if max_angle >= 90 and max_angle < 180:
                    obtuse_count += 1

            warnings.filterwarnings("default")
            answer = obtuse_count + 4

        ternary_answers.append(answer)

    # Create tuple for each relational question type and its respective answers
    ternary_relations = (ternary_questions, ternary_answers)
    binary_relations = (binary_questions, binary_answers)
    norelations = (norel_questions, norel_answers)
    
    # Create a tuple of the image along with its ground truth answers and return the dataset
    dataset = (STATE, ternary_relations, binary_relations, norelations)
    return dataset

# Bifurcate test and train datasets 
print('building test datasets...')
test_datasets = [build_dataset() for _ in range(test_size)]
print('building train datasets...')
train_datasets = [build_dataset() for _ in range(train_size)]


#img_count = 0
#cv2.imwrite(os.path.join(dirs,'{}.png'.format(img_count)), cv2.resize(train_datasets[0][0]*255, (512,512)))

# Store the train and test datasets into the declared directory
print('saving datasets...')
filename = os.path.join(dirs,'sort-of-clevr-state.pickle')
with open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets), f)
print('datasets saved at {}'.format(filename))