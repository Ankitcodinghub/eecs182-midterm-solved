# eecs182-midterm-solved
**TO GET THIS SOLUTION VISIT:** [EECS182 Midterm Solved](https://www.ankitcodinghub.com/product/eecs182-solved-7/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;116355&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EECS182 Midterm Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Exam location: Dwinelle 155

PRINT your student ID:

PRINT AND SIGN your name: ,

PRINT your discussion section: Row Number (front row is 1):

Name and SID of the person to your left:

Name and SID of the person to your right:

Do not turn this page until the proctor tells you to do so. You can work on Section 0 above before time starts.

Name and SID of the person in front of you:

Name and SID of the person behind you:

Section 0: Pre-exam questions (5 points)

2. What’s your favorite thing about this semester? (4 points)

3. Convolutional Neural Networks (19 points)

(a) (6 pts) Consider a convolutional neural network layer with:

• Input shape: [10, 3, 32, 32] (batch size, number of input channels, input height, input width)

• Number of filters: 64

For each of the following configurations of kernel size, stride, and padding, calculate the output dimensions [n,c,h,w] where n is the batch size, c is the number of channels, and h,w are the output height and width. Assume that each layer has the same input shape and number of filters as specified above.

i. Kernel size: 3×3, stride: 1, padding: 1

ii. Kernel size: 4×4, stride: 2, padding: 0

(b) (3 pts) Design a 3×3 filter that detects vertical edges like the one shown in the image below.

 

__ __ __

__ __ __

 

__ __ __

(c) (3 pts) Design a 3×3 filter to blur an image.

(Hint: blurring involves averaging a pixel’s value with those of its neighbors.)

 

__ __ __

__ __ __

 

__ __ __

PRINT your name and student ID:

(d) (7 pts) In a regular convolutional operation, the kernel slides over the input data in a contiguous manner. However, in dilated convolution, the kernel is “dilated” by introducing gaps between its elements, resulting in a larger receptive field for each output pixel.

Figure 1: Dilated convolution kernels. Source: “Brain MRI Super-Resolution Using 3D Dilated Convolutional Encoder–Decoder Network” by Du et al.

The dilation (d) determines the spacing between kernel elements — a dilation of d introduces d − 1 gaps between two kernel elements. The example above illustrates a 3×3 1-dilated kernel (d = 1 is equivalent to a regular convolutional kernel) and a 3×3 2-dilated (d = 2) kernel.

i. You are given an input matrix M and 2×2 filter k below. Compute their dilated convolution

with d = 2. Assume that gaps in the kernel are filled with zeros, stride is 1 and padding is 0.

 

1 2 3

M = 4 5 6

 

7 8 9

ii. Consider a two-layer architecture:

DilatedConv1(3×3, d=1) → DilatedConv2(3×3, d=2)

Both layers use the same sized kernel (3×3), but DilatedConv1 has a dilation d = 1 and DilatedConv2 has a dilation d = 2. Compute the size of the receptive field at the output of the final layer (DilatedConv2).

Recall that the receptive field is the region in the original input image whose pixels affect the output for a pixel in the specified layer.

4. Beam Search (8 points)

Consider performing beam search with beam size k = 2 that expands the top k sequences until an &lt;EOS&gt; token (end of sequence) is reached. Upon keeping a completed sequence as one of the top-k, the beam size for continuing the remaining search decreases by 1.

The log-probabilities of each word at a given timestep are shown next to the word.

What are the 2 completed sequences that this decoder would consider?

What are their overall log-probabilities?

[Extra page. If you want the work on this page to be graded, make sure you tell us on the problem’s main page.]

5. Attention on Linear RNNs (14 points)

Consider a linear RNN with a two-dimensional state ht and a scalar input ut at each timestep:

ht = Whht−1 + Wuut

Suppose , and we initialize h . The inputs are u1 = 1, u2 = 2,

u3 = −1. The decoder state is hout = Whh3.

We also generate the keys, values, and queries for decoder-side cross-attention as purely linear functions of the state using weight matrices:

A partially filled in computation graph is provided below. The edges with weights W on them indicate that the vector is left-multiplied by that weight matrix as it passes along that edge.

(a) (10 pts) Fill in the blanks on the diagram above for the missing hidden states, keys, and values.

(Hint: You can also use some of the given numbers to check your work.)

(b) (4 pts) What would be the output of attention for the decoder’s query? To simplify calculations, use an argmax instead of softmax. For example, softmax([1,3,2]) becomes argmax([1,3,2]) = [0,1,0]. [Extra page. If you want the work on this page to be graded, make sure you tell us on the problem’s main page.]

6. Debugging neural networks (21 points)

Which initialization methods are A, B, and C, (circle your choice) and very briefly explain your reasoning below it.

A is Zero Xavier He B is Zero Xavier He

C is Zero Xavier He

PRINT your name and student ID:

(b) (12 pts) You are designing a neural network to perform image classification. You want to use data augmentation to regularize the training for your model. You write the following code:

import random from torch.utils.data import DataLoader, Subset

train_dataset = load_train_dataset()

augmented_dataset = apply_augmentations(train_dataset)

num_data = len(augmented_dataset) indices = list(range(num_data)) random.shuffle(indices) split = int(0.8 * num_data) train_idxs, val_idxs = indices[:split], indices[split:]

train_data = Subset(augmented_dataset, train_idxs) val_data = Subset(augmented_dataset, val_idxs) test_data = load_test_dataset()

train_loader = DataLoader(train_data, batch_size=32) val_loader = DataLoader(val_data, batch_size=32) test_loader = DataLoader(test_dataset, batch_size=32)

# Train the model for epoch in range(10):

for images, labels in train_loader: # Training code here

# Validation code here

# Testing code here

On running this code, you find that you get a high training accuracy and validation accuracy, but a significantly lower testing accuracy as compared to the validation accuracy.

7. Learning from Point Clouds (22 points)

A point cloud is a discrete set of data points in space. Because of this set-valued nature of point clouds, concepts from graph neural nets are often relevant in their processing.

In Fig.2, we consider a simple network to process a 2d point cloud , where n is the number of points. The original features of each point are its horizontal and vertical coordinates.

𝑌+(𝑘)

Input point cloud Point feature learning Global feature 𝐹 score Output 𝑌#

Figure 2: 2d point cloud processing network.



0

0

For example, an input point cloud with ground-truth of digit 1 could be represented as  0



0 

4

3

 where each

2



1

row is a different point in the point cloud.

(a) (8 pts) The point feature learning module in Fig.2 learns f-dimensional features for each point separately. Specifically, it learns (shared) weights W1 ∈ R2×f to get hidden layer outputs Z = XW1. We then apply the nonlinear activation function element-wise and then use average pooling to yield the f-dimensional global feature vector F ∈ Rf.

Suppose we swap the first two points of the input point cloud X, i.e. {x1,x2,…,xn} to {x2,x1,…,xn}. Show that the global feature F will not change.

Note: In reality, the network here is permutation invariant, as changing the ordering of the n input points in X will not affect the global feature F.

PRINT your name and student ID:

(b) (8 pts) One drawback of the point feature learning discussed in part (a) is that the spatial interrelationships of different points is not considered.

One idea is to use the Euclidean distance as a similarity measure to group points together into local neighborhoods, and to then process each point together with contextual information about that neighborhood. Your friend proposed several different point grouping methods listed below.

Select all methods that guarantee permutation-invariance, i.e. the global feature vector F will not change with different orderings of the points.

For each point xi of the point cloud X, we find the top-m nearest neighbor points. We then augment each point coordinates with its nearest neighbors’ coordinates to make Xe ∈ Rn×2(m+1). The order of a concatenated group of points would be: the center point, 1st closest, 2nd closest point, …, mth closest point. Now where are the shared learnable weights.

For each point xi of the point cloud X, we find the top-m nearest neighbor points. As in the previous choice, we then augment each point coordinates with its nearest neighbors’ coordinates and get Xe ∈ Rn×2(m+1). The difference from the previous choice is that the order of a concatenated group of nearby points simply follows their order in the original X. The is the same as the previous choice.

For each point xi of the point cloud X, we instead find all neighboring points with the distance to xi smaller than a predefined radius r. We then augment each point coordinates with its neighbors’ coordinates with the order being the center point, the 1st closet point within r, the 2nd closet point within r, …, the furthest point within r. Because different points might have a different number of neighbors within radius r, the shared learnable weights Wf1 ∈ R2(n+1)×f are applied by using the relevant-size truncation of Wf1 for every point.

For each point xi of the point cloud X, as in the previous option, we find all neighboring points with the distance to xi smaller than a predefined radius r. But instead of concatenating the representation of the point with its neighboring points, we instead extend the point’s representation with just the distance d from xi to the furthest point within the radius r resulting in an Xe ∈ Rn×3. The Wf1 ∈ R3×f.

PRINT your name and student ID:

(c) (6 pts) Point downsampling (reducing the number of points for deeper layers to process). Let’s consider a deeper network, as shown in Fig.3. We are adding a pooling layer (highlighted in bold text) after the first point feature learning layer to downsample half of the points in the cloud (Z → Z1), followed by another point feature learning layer to increase the dimension of the point features from f to 2f (Z1 → Z2). (Note that this mimics pooling procedures in CNNs: the spatial size shrinks, followed by an increase of the feature dimensionality.)

Input point cloud Point feature learning 1 Point feature learning 2 Global feature 𝐹 score Output 𝑌#

Figure 3: A deeper point cloud processing network.

Consider two candidate algorithms. Both start with the point cloud comprising n points and both iteratively select points until you have at least samples. For both, we construct two sets: sampled and remaining to denote the set of sampled and remaining points. For both, we first pick a random point and use it to initialize sampled and we initialize remaining with all the other points. Then, the iterative processes are different for the two algorithms as described below.

Which algorithm is more similar in spirit to using standard max pooling for downsampling in CNNs?

◦ Algorithm 1:

i. For each point in remaining find its nearest neighbor in sampled, saving the distance.

ii. Select the point in remaining whose nearest neighbor distance is the largest and move it from remaining to sampled. (i.e. We keep points far from those we already have.) ◦ Algorithm 2:

i. For each point in remaining find its nearest neighbor in sampled, saving the distance.

ii. Select the point in remaining whose nearest neighbor distance is the smallest and move it from remaining to sampled. (i.e. We keep points close to those we already have.) At the end, only the points in sampled get sent on to the next layer.

8. Normalization (14 points)

(a) (2 pts) Consider the following digram where the shaded blocks are the entries participating in one normalization step for a CNN-type architecture. N represents the mini-batch, H,W represent the different pixels of the “image” at this layer, and C represents different channels.

• Which one denotes the process of batch normalization? Circle your selection below.

A B C

• Which one denotes layer normalization? Circle your selection below.

A B C

(b) (12 pts) Consider a simplified BN where we do not divide by the standard deviation of the data batch. Instead, we just de-mean our data batch before applying the scaling factor γ and shifting factor β. For simplicity, consider scalar data in an n-sized batch: [x1,x2,…,xn]. Specifically, we let xˆi = xi − µ where µ is the average across the batch and output [y1,y2,…,yn] where yi = γxˆi + β to the next layer. Assume we have a final loss L somewhere downstream. Calculate in terms of for j = 1,…,n as well as γ and β as needed.

Numerically, what is when n = 1 and our input batch just consists of [x1] with an output batch of [y1]? (Your answer should be a real number. No need to justify.)

What happens when n → ∞? (Feel free to assume here that all relevant quantities are bounded.)

9. Optimizers (10 points)

Algorithm 1 SGD with Momentum Algorithm 2 Adam Optimizer (without bias correction)

1: Given η = 0.001,β1 = 0.9,β2 = 0.999

2: Initialize time step t ← 0, parameter θt=0 mt=0 ← 0, vt=0 ← 0

3: Repeat

7:

9: Until the stopping condition is met ∈ Rn,

1: Given η = 0.001,β1 = 0.9

2: Initialize:

3: time step t ← 0

4: parameter θt=0 ∈ Rn

5: Repeat

6: t ← t + 1

7: gt ← ∇ft(θt−1)

8: mt ← β1mt−1 + (1 − β1)gt

9: θt ← θt−1 − ηmt

10: Until the stopping condition is met

(a) (4 pts) Complete part (A) and (B) in the pseudocode of Adam.

(b) (6 pts) This question asks you to establish the relationship between

• L2 regularization for vector-valued weights θ refers to adding a squared Euclidean norm of the weights to the loss function itself:

• Weight decay refers to explicitly introducing a scalar γ in the weight updates assuming loss f:

θt+1 = (1 − γ)θt − η∇f(θt)

where γ = 0 would correspond to regular SGD since it has no weight-decay.

Show that SGD with weight decay using the original loss ft(θ) is equivalent to regular SGD on the L2-regularized loss ftreg(θ) when γ is chosen correctly, and find such a γ in terms of λ and η.

10. Self-Supervised Linear Purification (31 points)

Consider a linear encoder — square weight matrix W ∈ Rm×m — that we want to be a “purification” operation on m−dimensional feature vectors from a particular problem domain. We do this by using selfsupervised learning to reconstruct n points of training data X ∈ Rm×n by minimizing the loss:

L1(W;X) = kX − WXk2F (1)

While the trivial solution W = I can minimize the reconstruction loss (1), we will now see how weightdecay (or equivalently in this case, ridge-style regularization) can help us achieve non-trivial purification.

(2)

Reconstruction Loss Regularization Loss

Note above that λ controls the relative weighting of the two losses in the optimization.

(a) (8 pts) Consider the simplified case for m = 2 with the following two candidate weight matrices:

(3)

The training data matrix X is also given to you as follows:

X (4) i. Compute the reconstruction loss and the regularization loss for the two encoders, and fill in the missing entries in the table below.

Encoder Reconstruction Loss Regularization Loss

α __________ __________

β 0.001 __________

ii. For what values of the regularization parameter λ is the identity matrix W(α) not a better solution for the objective L2 in (2), as compared to W(β)?

[Extra page. If you want the work on this page to be graded, make sure you tell us on the problem’s main page.]

(b) (15 pts) Now consider a generic square linear encoder W ∈ Rm×m and the regularized objective L2 reproduced below for your convenience:

Reconstruction Loss Regularization Loss

Assume σ1 &gt; ··· &gt; σm ≥ 0 are the m singular values in X, that the number of training points n is larger than the number of features m, and that X can be expressed in SVD coordinates as X = UΣV&gt;.

i. You are given that the optimizing weight matrix for the regularized objective L2 above takes the

following form. Fill in the empty matrices below.



Wc = . (5)

ii. Derive the above expression.

(Hint: Can you understand L2(W;X,λ) as a sum of m completely decoupled ridge-regression problems?)

(c) (8 pts) You are given that the data matrix X ∈ R8×n has the following singular values:

{σi} = {10,8,4,1,0.5,0.36,0.16,0.01}

For what set of hyperparameter values λ can we guarantee that the learned purifier Wc will preserve at least 80% of the feature directions corresponding to the first 3 singular vectors of X, while attenuating components in the remaining directions to at most 50% of their original strength?

(Hint: What are the two critical singular values to focus on?)

PRINT your name and student ID:

[Doodle page! Draw us something if you want or give us suggestions or complaints. You can also use this page to report anything suspicious that you might have noticed.

If needed, you can also use this space to work on problems. But if you want the work on this page to be graded, make sure you tell us on the problem’s main page.]
