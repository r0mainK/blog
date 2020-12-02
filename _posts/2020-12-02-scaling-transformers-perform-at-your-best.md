---
layout: post
title: "Scaling Transformers: Perform at your best"
date: 2020-12-02
katex: true
---
Okay, so it took a bit longer to wrap up than I expected ... but here is the second part of my _Scaling Transformers_ series 🤗 If you didn't catch the first part in which I talked about the Reformer, you can find it [over here](/2020/11/24/scaling-transformers-reform-your-ways.html). As you may recall, in the previous part I decided not to implement one of the key innovations of the model: the LSH-based approximation of _attention_. The reason is simple: a couple months ago a new approach was introduced, and clearly outperformed previous work. So, without further ado let's jump right in, and talk about ...


## The Performer

This second paper I've been referencing is [Rethinking Attention with Performers](https://arxiv.org/pdf/2009.14794.pdf), a joint effort by Google, the University of Cambridge, DeepMind, and the Alan Turing Institute. As mentioned earlier, they propose a method for approximating attention. Before going further, you may wonder why we would even need to approximate something as easily computable as attention ? I've touched upon this previously and the answer is quite simple: computing attention over a sequence scales quadratically with said sequence's length, making it memory-intensive on long sequences - and more generally, a performance bottleneck.

Previous work such as the Reformer or the [Linformer](https://arxiv.org/pdf/2006.04768.pdf) had already proposed schemes to alleviate the problem, but as the authors of the Performer noted, these methods relied on strong priors on the input sequences, assuming the attention matrix was sparse or of low-rank - and lacked strong guarantees on the accuracy of the resulting representations. Killing two birds with one stone, the authors proposed a method which scales linearly with sequence length, and comes with strong theoretical guarantees on the approximation quality 🤩

As the paper is pretty math-heavy I'll let you review the proof to their claims for yourself if you are interested, and focus on explaining and implementing the approximation algorithms. Furthermore, while they provided both a causal and a non-causal version of the algorithm, I will be focusing on the non-causal one, since my focus with this project is towards masked language modeling. You can check out the [article](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html) Google released for more information on the subject, or the paper.


### Faster attention

{% katexmm %}
As a reminder, let's go over how attention is computed. Given a sequence length $L$ and embedding dimension $d$, query, key and value matrices of dimension $(L,d)$ are somehow produced. In the case of _self-attention_ for instance, the input sequence is just mapped using linear layers. Denoting $\mathbf{q}_n$, $\mathbf{k}_n$ and $\mathbf{v}_n$ the query, key and value vectors (of dimension $d$) for position $n$, then output for position $i$ is computed as follows:
{% endkatexmm %}

{% katex display %}
\mathbf{Out}_i = \sum_{j=1}^L \frac{\exp(\dfrac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d}})}{ \sum_{l=1}^L\exp(\dfrac{\mathbf{q}_i^T \mathbf{k}_l}{\sqrt{d}})} \mathbf{v}_j
{% endkatex %}

{% katexmm %}
For efficiency rows are compacted in matrices $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ of dimension $(L,d)$, so the output for all positions is computed in one go:
{% endkatexmm %}

{% katex display %}
\mathbf{A} = \exp(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}})\quad
\mathbf{D} = diag(\mathbf{A}\mathbf{1}_L)\quad
\mathbf{Out} = \mathbf{D}^{-1}\mathbf{A}\mathbf{V}
{% endkatex %}

{% katexmm %}
The $O(L^2)$ complexity of attention comes from the product of the query and key matrices. To attenuate it, one possible way would be to split $\mathbf{A}$ somehow, and avoid that multiplication. Well, that's exactly what the authors proposed, by transforming $\mathbf{Q}$ and $\mathbf{K}$ through _random feature maps_. I know, the name of this method seems a bit abstract ... but I promise it's not that hard to understand 🤞

For the time being, let's assume such a feature map is applied to $\mathbf{Q}$ and $\mathbf{K}$, yielding $\mathbf{Q}'$ and $\mathbf{K}'$ of dimensions $(L,r)$. The computations can now be done in a different order, avoiding the quadratic complexity:
{% endkatexmm %}

{% katex display %}
\mathbf{D}' = diag(\mathbf{Q}'({\mathbf{K}'}^T\mathbf{1}_L))\quad
\mathbf{Out} = {\mathbf{D}'}^{-1} (\mathbf{Q}'({\mathbf{K}'}^T\mathbf{V}))
{% endkatex %}

{% katexmm %}
If you're paying ... _attention_, you should see there's something off with this 🤔 Since the diagonal matrix has dimension $(L, L)$, and the second term has dimension $(L, d)$, the computation should still scale quadratically, right ? Not quite, although it isn't immediately clear why.

When you multiply a matrix with a diagonal matrix, you're really just rescaling it, i.e. $[\mathbf{D}\mathbf{X}]_{i,j} = d_i\mathbf{X}_{i,j}$. Although such an operation is not directly built into Pytorch, the [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html) function can be used to create it - you don't even need to create the diagonal matrix, the diagonal itself is enough 🤗 

Here is how I implemented it, given $\mathbf{Q}'$, $\mathbf{K}'$ and $\mathbf{V}$:
{% endkatexmm %}

```python
def apply_scaling(scale, x):
    return torch.einsum("...n,...nd->...nd", scale, x)

def fast_attention(query, key, value):
    buffer = torch.cat([key.transpose(1, 2).bmm(value), key.sum(1).unsqueeze(-1)], dim=-1)
    buffer = query.bmm(buffer)
    return apply_scaling(1 / buffer[:, :, -1], buffer[:, :, :-1])
```

{% katexmm %}
The only notable thing here is that this function operates on batches of sequences, so batch multiplication is used instead of the regular one (tensors are batched along the first dimension). The above code has time and space complexity of $O(Lrd)$ and $O(Ld + Lr + rd)$ per sequence, versus $O(L^2d)$ and $O(Ld + L^2)$ for regular attention - so with a small enough $r$, the performance gains should be quite important on long sequences.
{% endkatexmm %}

Now, let's see how to use this, by talking a bit about ...

### Feature maps

{% katexmm %}
So what are these maps ? Well, dear reader, you might know of the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick). Given a kernel function $K: (\mathcal{X},\mathcal{X}) \rightarrow \mathbb{R}$ which verifies some properties, it states the existence of a feature map $\phi: \mathcal{X} \rightarrow \mathcal{V}$ for which:
{% endkatexmm %}

{% katex display %}
K(\mathbf{x}, \mathbf{y}) =  \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle_\mathcal{V}
{% endkatex %}

{% katexmm %}
If it's your first encounter with the trick, you may be wondering how to choose $\mathcal{V}$ and $\phi$, or how one would use it. Without going too much into detail, the key idea is that you don't need an explicit representation of the feature map, the fact that it exists is often enough to create mathematical shortcuts in learning algorithms. For instance, it allows [SVMs](https://en.wikipedia.org/wiki/Support_vector_machine#Kernel_trick)s to learn non-linear functions.

Okay great, but what has this got to do with attention ? Well, notice that each element in $A$ can be seen as:

{% endkatexmm %}
{% katex display %}
\mathbf{A}_{i,j} = \exp(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d}}) =  \exp(\frac{\mathbf{q}_i^T}{\sqrt{\sqrt{d}}}\frac{\mathbf{k}_j}{\sqrt{\sqrt{d}}})  = \exp(\tilde{\mathbf{q}_i}^T\tilde{\mathbf{k}_j}) =  K(\tilde{\mathbf{q}_i}, \tilde{\mathbf{k}_j})
{% endkatex %}

{% katexmm %}
With $K: (\mathbb{R}^d,\mathbb{R}^d) \rightarrow \mathbb{R}$ the _softmax_ kernel and $\tilde{\mathbf{q}_i}$, $\tilde{\mathbf{k}_j}$ the renormalized query and keys. This kernel fulfills the conditions of the trick, so the feature maps exist. Now, as I said earlier the explicit representations of feature maps are never used, so as is this doesn't change much ... however, I left out a part of the story 😊

Coming back to SVMs, it turns out that even with the kernel trick, things weren't peachy. Take a classification problem with $N$ data points in dimension $d$. In order to evaluate the learned machine at a single point $\mathbf{x}$, one has to compute $f(\mathbf{x})=\sum_{i=1}^N c_i K(\mathbf{x},\mathbf{x_i})$, where $c_i$ are learned parameters. This scales in $O(Nd)$, which is pretty bad for large datasets since you are going to evaluate the machine many times before getting to a solution.

To circumvent this issue, a novelty was proposed: instead of relying on the implicit mapping provided by the kernel trick, let's go one step further and directly approximate the inner product, by mapping points to a low dimensional space with a _randomized feature map_ $\mathbf{z}$:
{% endkatexmm %}

{% katex display %}
K(\mathbf{x}, \mathbf{y}) =  \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle_\mathcal{V} \approx \mathbf{z}(\mathbf{x})^T\mathbf{z}(\mathbf{y})
{% endkatex %}

{% katexmm %}
And this solves the problem 👏 Indeed there is no more need to use a complex algorithm like SVM, since a linear algorithm applied to the transformed points can do the job: $f(\mathbf{x}) = \mathbf{W}^T\mathbf{z}(\mathbf{x})$ ! This may seem surprisingly simple, but it turns out this approach works great - given the right maps.

With this in mind, you can imagine what the next step is: look into what exactly these random feature maps are, find one adequate to the problem, and use it to transform $\mathbf{Q}$ and $\mathbf{K}$ !
{% endkatexmm %}

#### **Random Fourier Features**

{% katexmm %}
In [the paper](http://www.cs.cornell.edu/selman/local/notes-new/pdf/2013_03_06_note_readings_randomization_vs_optimization_machine_learning_neural_nets_rahimi-recht-random-features.pdf) which first introduced these concepts, the authors show that for certain kernels, you can use the [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to get a random feature map. If you think this is getting a bit complex ... well you're not wrong 😅 But no worries, it's not that hard to understand. The class of kernels for which their method works are shift-invariant positive-definite kernels:
{% endkatexmm %}

{% katex display %}
\forall (\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d, \quad K(\mathbf{x}, \mathbf{y})=K(\mathbf{x} - \mathbf{y}) \geq 0
{% endkatex %}

{% katexmm %}
Since the kernel is shift-invariant it's only a function of one parameter, so it's possible to define $\mathbf{p}$ it's Fourier transform, and rewrite:
{% endkatexmm %}

{% katex display %}
K(\mathbf{x}, \mathbf{y}) = K(\mathbf{x} - \mathbf{y}) = \int_{\mathbb{R}^d} \mathbf{p}(\omega)e^{j\omega^T(\mathbf{x} - \mathbf{y})}d\omega
{% endkatex %}

{% katexmm %}
The main proposition backing their method up is Bochner's theorem which states that such a kernel can be positive-definite if and only if its Fourier transform is a proper probability distribution. If $\mathcal{D}$ is said distribution, then since the kernel is real-valued:
{% endkatexmm %}

{% katex display %}
\begin{aligned}
K(\mathbf{x}, \mathbf{y}) &=  \int_{\mathcal{D}} e^{j\tilde{\omega}^T(\mathbf{x} - \mathbf{y})}d\tilde\omega \\
&= \mathbb{E}_{\omega \sim \mathcal{D}}[{e^{j\omega^T(\mathbf{x} - \mathbf{y})}}] \\
&= \mathbb{E}_{\omega \sim \mathcal{D}}[{\cos{\omega^T(\mathbf{x} - \mathbf{y})}}]
\end{aligned}
{% endkatex %}

{% katexmm %}
With $z_\omega(\mathbf{x}) = [\cos \omega^T \mathbf{x}, \sin \omega^T \mathbf{x}]$ and the use of trigonometry, the random feature map becomes visible:
{% endkatexmm %}

{% katex display %}
\begin{aligned}
K(\mathbf{x}, \mathbf{y}) &= \mathbb{E}_{\omega \sim \mathcal{D}}[\cos \omega^T \mathbf{x} \cos \omega^T \mathbf{y} + \sin\omega^T \mathbf{x} \sin \omega^T \mathbf{y}] \\ &= \mathbb{E}_{\omega \sim \mathcal{D}}[z_\omega(\mathbf{x})^Tz_\omega(\mathbf{y})^T]
\end{aligned}
{% endkatex %}

{% katexmm %}
The kernel can now be approximated by simply sampling $\omega$ from $\mathcal{D}$. To lower the estimator's variance, the sampling is done $m$ times and the corresponding $z_{\omega_i}$ are stacked in vectors $\mathbf{z}$, and rescaled by a factor $\frac{1}{\sqrt{m}}$. With this, the approximation is clear:
{% endkatexmm %}

{% katex display %}
K(\mathbf{x}, \mathbf{y}) \approx \mathbf{z}(\mathbf{x})^T \mathbf{z}(\mathbf{y}) =  \frac{1}{m}\sum_{i=1}^m \cos \omega_i^T (\mathbf{x} - \mathbf{y})
{% endkatex %}

{% katexmm %}
It turns out that the softmax kernel is both shift-invariant and positive-definite, so to use this results one would simply need to get to $\mathcal{D}$ via the Fourier transform - _which is not that easy_. However, using this identity:
{% endkatexmm %}

{% katex display %}
\lVert \mathbf{x} - \mathbf{y} \rVert^2 = \lVert \mathbf{x} \rVert ^2   + \lVert \mathbf{y} \rVert^2  - 2 \mathbf{x}^T \mathbf{y}
{% endkatex %}

The softmax kernel can be rewritten as function of the Gaussian kernel, which is also shift-invariant and positive-definite:

{% katex display %}
K(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^T \mathbf{y})= \exp(\frac{\lVert \mathbf{x} \rVert ^2}{2})K_{gauss}(\mathbf{x}, \mathbf{y})\exp(\frac{\lVert \mathbf{y} \rVert^2 }{2})
{% endkatex %}

{% katexmm %}
It's well known that the Fourier transform of a Gaussian is a Gaussian ... which means that by sampling random vectors from the normal distribution $\mathcal{N}(0,\mathbb{I}_d)$, the random Fourier feature map described above can be used to approximate the Gaussian kernel, finally yielding:
{% endkatexmm %}

{% katex display %}
K(\mathbf{x}, \mathbf{y}) \approx \exp(\frac{\lVert \mathbf{x} \rVert ^2}{2})\mathbf{z}(\mathbf{x})^T \mathbf{z}(\mathbf{y}))\exp(\frac{\lVert \mathbf{y} \rVert^2 }{2}) = \tilde{\mathbf{z}}(\mathbf{x})^T\tilde{\mathbf{z}}(\mathbf{y})
{% endkatex %}

Now all that's left to approximate attention is to implement this 🥳 


#### **Positive Random Features** 

Okay so, long story short ... _I lied_ 😈

{% katexmm %}
While everything that I described in the previous section is true, and should help you understand this section, random Fourier features can't be used. As the authors of the Performer paper show, the issue is that these features approximate the softmax kernel in a way that may yield negative values, especially when the true output of the kernel is close to zero - as is often the case with attention since most tokens are irrelevant. When attention scores are renormalized via $D^{-1}$, negative scores will result in abnormalities, such as negative diagonal values 😱
{% endkatexmm %}

This lead the authors to introduce **positive** random features. How did they do it ? Well, I won't cover everything as the proof is included in the paper, but the process is similar to what was done in the previous section. Using the following identity:

{% katex display %}
\lVert \mathbf{x} + \mathbf{y} \rVert^2 = \lVert \mathbf{x} \rVert ^2   + \lVert \mathbf{y} \rVert^2  + 2 \mathbf{x}^T \mathbf{y}
{% endkatex %}

The softmax kernel can be rewritten as:

{% katex display %}
K(\mathbf{x}, \mathbf{y}) = \exp(-\frac{\lVert \mathbf{x} \rVert ^2}{2})\exp(\frac{\lVert \mathbf{x} + \mathbf{y} \rVert^2}{2})\exp(-\frac{\lVert \mathbf{y} \rVert^2 }{2})
{% endkatex %}

This is where things change. The middle term is obviously not shift-invariant, so the Fourier transform can't be used here. However, the authors were able to prove that:

{% katex display %}
\exp(\frac{\lVert \mathbf{x} + \mathbf{y} \rVert^2}{2}) = \mathbb{E}_{\omega \sim \mathcal{N}(0,\mathbb{I}_d)}[\exp(\omega^T\mathbf{x}) \exp(\omega^T(\mathbf{y})]
{% endkatex %}

{% katexmm %}
And that's it ! Since the exponential function is positive, it's clear that the approximated values will never be positive, meaning the issue is solved 🤩 From here the method is the same, $m$ random features are sampled from $\mathcal{N}(0,\mathbb{I}_d)$, and the random feature map can be defined as:
{% endkatexmm %}

{% katex display %}
z_{w_i} = \exp(\omega_i^T\mathbf{x} - \frac{\lVert \mathbf{x} \rVert ^2}{2})\quad \mathbf{z}(\mathbf{x}) = \frac{1}{\sqrt{m}}[z_{w_1}, \dots, z_{w_m}]
{% endkatex %}

It turns out an estimator with even lower variance can be used, by doubling the size of the map (but with the same amount of random vectors). Indeed, using the fact that:

{% katex display %}
\exp\omega^T(\mathbf{x} + \mathbf{y}) = \frac{1}{\sqrt{2}}(\exp (\omega^T\mathbf{x})\exp (\omega^T\mathbf{y}) + \exp(-\omega^T\mathbf{x})\exp (-\omega^T\mathbf{y}))
{% endkatex %}

A second random feature map can be deduced (the authors call it _hyperbolic_ as it's derived from the hyperbolic cosine function):

{% katex display %}
\begin{cases}
z_{w_i} = \frac{1}{\sqrt{2}}\exp(\omega_i^T\mathbf{x} - \frac{\lVert \mathbf{x} \rVert ^2}{2}) \newline \\
\tilde{z}_{w_i} = \frac{1}{\sqrt{2}}\exp(-\omega_i^T\mathbf{x} - \frac{\lVert \mathbf{x} \rVert ^2}{2})
\end{cases} \quad \mathbf{z}(\mathbf{x}) = \frac{1}{\sqrt{m}}[z_{w_1}, \dots, z_{w_m}, \tilde{z}_{w_1}, \dots, \tilde{z}_{w_m}]
{% endkatex %}

Both these feature maps are straightforward to code:

```python
def apply_regular_feature_map(x, orf, epsilon=1e-6):
    m, d_k = orf.shape
    proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
    norm = (x ** 2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
    return (torch.exp(proj_x - norm) + epsilon) / math.sqrt(m)


def apply_hyperbolic_feature_map(x, orf, epsilon=1e-6):
    m, d_k = orf.shape
    proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
    proj_x = torch.cat([proj_x, -proj_x], dim=-1)
    norm = (x ** 2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
    return (torch.exp(proj_x - norm) + epsilon) / math.sqrt(2 * m)

```

{% katexmm %}
So, a couple things:
- the embedding dimension is named $d_k$ because attention will be done via multiple heads, _more on that soon_
- as I explained earlier the input is renormalized by $d_k^{-\frac{1}{4}}$
- mimicking the released Jax code, a small $\epsilon$ is added for stability

There are still some things to address, but the hard-lifting is now done 😎 Before moving on though, I'd like to point one last thing.

If you check out the original Jax code, you will find a slight difference here. Indeed, an additional term is substracted before exponentiation, namely the maximum of `proj_x`. Surprisingly, the way they compute the maximum for queries and keys differs: it's taken over all values in the sequence for the former,   and across values of **all** sequences in the batch for the latter 🤔 This is probably some stability-inducing hack, as the term is canceled after normalization. Since I have no reason to believe it's necessary, I removed it, so be warned !
{% endkatexmm %}

### Loose ends

As I mentioned, there are still some details that need to be covered. Let's start with the hardest - don't worry, it shouldn't be anything you can't handle if you made it this far 🤗

#### **Orthogonal Random Features**

It turns out that an additional way to increase the quality of the approximation is by having the sampled random vectors that form an orthogonal base, i.e. if:

{% katex display %}
\forall (i,j), \quad \omega_i^T \omega_j = \mathbf{1}_{i=j}
{% endkatex %}

{% katexmm %}
To do so, the method presented in [this paper](https://arxiv.org/pdf/1610.09072v1.pdf) can be employed. It relies on [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition), in which a square matrix is decomposed in an orthogonal matrix $\mathbf{Q}$ and an upper triangular matrix $\mathbf{R}$. Given that the matrix of random features $\Omega$ is of dimension $(m, d)$, the decomposition may not be possible if $m \neq d$. In that case:

- if $d > m$ then more vectors are sampled to get to $d$
- if $d < m$ then the matrix is divided into $k$ blocks $\Omega_k$ of dimensions $(d,d)$, and if the last block is too small more vectors are sampled

Afterwords each block is decomposed and the orthogonal matrices $\mathbf{Q}_k$ are concatenated. According to the authors of this method, the resulting matrix should still be orthogonal with high probability. However, the rows obtained after QR-decomposition have unit-norm while the ones sampled from a normal distribution follow a $\chi$-distribution, so the matrix needs to be rescaled. It's a pretty simple procedure, as the sum of squared normal random vectors also follow that distribution. 

Here is my implementation of the whole process (I reused the `apply_scaling` function defined previously):

{% endkatexmm %}

```python
def create_orf(d_k, m):
    blocks = torch.randn(math.ceil(m / d_k), d_k, d_k)
    blocks, _ = torch.qr(blocks)
    scale = torch.randn(m, d_k).norm(dim=1)
    return apply_scaling(scale, blocks.reshape(-1, d_k)[:m])
```

#### **Multiple headed module**

Transformer models often have multiple headed attention blocks, rather than just one. This is not big deal, but I've completely glossed over this until now, so let's see how to handle it ! Here is my implementation of the attention block:

```python
class FastSelfAttention(Module):
    def __init__(self, d_model, h, m, use_hyperbolic):
        super(FastSelfAttention, self).__init__()
        self.h = h
        self.linears = ModuleList([Linear(d_model, d_model) for _ in range(4)])
        self.register_buffer("orf", create_orf(d_model // h, m), persistent=False)
        self.apply_feature_map = apply_regular_feature_map
        if use_hyperbolic:
            self.apply_feature_map = apply_hyperbolic_feature_map

    def redraw_orf(self):
        m, d_k = self.orf.shape
        orf = create_orf(d_k, m)
        orf = orf.to(self.orf.device)
        self.register_buffer("orf", orf, persistent=False)

    def split_by_head(self, x, B, L):
        return x.view(B, L, self.h, -1).permute(0, 2, 1, 3).reshape(B * self.h, L, -1)

    def concat_by_head(self, x, B, L):
        return x.reshape(B, self.h, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

    def forward(self, x):
        B, L, _ = x.shape
        query, key, value = (self.split_by_head(l(x), B, L) for l in self.linears[:3])
        query = self.apply_feature_map(query, self.orf)
        key = self.apply_feature_map(key, self.orf)
        out = fast_attention(query, key, value)
        out = self.concat_by_head(out, B, L)
        out = self.linears[3](out)
        return out
```

{% katexmm %}
For a given sequence, each head is applied on a specific part of the embedding of dimension `d_model // h` where `h` is the number of heads. Since the heads are independent, in effect it's as if `B * h` separate sequences are being processed, before the output linear projection. If you understand this, then most of the code should be clear. To recap:

- the orthogonal random feature matrix `orf` has dimension `(m, d_model // h)`, since each part of the sequence will be projected separately
- the query, key and value matrices are reshaped, so that they have dimension `(B *h, L, d_model // h)` - a custom function is needed so that the reshaping is done correctly
- the output is also reshaped to have dimension `(B, L, d_model)` by a custom function

By the way, if you're wondering how to pick $m$, the authors showed in the paper that a good value should be around $d\log(d)$, with $d$ the heads' embedding dimension. For BERT-base, the model has 12 heads and a total embedding dimension of 768, thus the heads' embedding dimension is 64. In their experiments, the authors chose $m=256$ to stay with powers of 2.

{% endkatexmm %}

## In conclusion

And that's all for this second part ! As expected, it was much more math-heavy then the first part, but I wanted to get into the backstory of this paper. Hopefully the somewhat useless tangent on Fourier didn't annoy you 😅 In all cases, if you want to check out the project you can find it on GitHub [here](https://github.com/r0mainK/outperformer) (stars are welcome) !

I'm planning on keeping this series going and working on a new article as soon as possible, but not sure when it will come, as December has started and with it the [Advent of Code](https://adventofcode.com/) 🎅

In all cases if you have any feedback please hit me up, and if you want me to look into a specific paper, please say so 😄

Finally, as last time I'd like to thank everybody that contributed to the resources below - you should check them out if you've got the time 👌

## Resources

### Code

- [Official Performer code](https://github.com/google-research/google-research/tree/master/performer/fast_self_attention): the code released by the authors of the Performer paper, in Jax, which I didn't end up using as much as I still don't know Jax 😅

- [performer-pytorch package](https://github.com/lucidrains/performer-pytorch): a more complete library implementing both the causal and non-causal attention from the Performer, as well as the Reformer paper. It was super useful when it came to testing my code, as the author of the library transpiled most of the Performer code from Jax to Pytorch. My project ended up looking quite different since my scope was smaller, so I definitely recommend checking it out 👌



### Papers



- [Rethinking Attention with Performers](https://arxiv.org/pdf/2009.14794.pdf), by Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell and Adrian Weller

- [Random Features for Large-Scale Kernel Machines](http://www.cs.cornell.edu/selman/local/notes-new/pdf/2013_03_06_note_readings_randomization_vs_optimization_machine_learning_neural_nets_rahimi-recht-random-features.pdf), by Ali Rahimi and Benjamin Recht

- [Orthogonal Random Features](https://arxiv.org/pdf/1610.09072v1.pdf), by Felix X. Yu, Ananda Theertha Suresh, Krzysztof Choromanski, Daniel Holtmann-Rice and Sanjiv Kumar
