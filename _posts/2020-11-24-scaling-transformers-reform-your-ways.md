---
layout: post
title: "Scaling Transformers: Reform your ways"
date: 2020-11-24
katex: true
---
After showing you my Python setup in the previous post, I wanted to showcase it in a second post with a project. However it got out of hand: I chose to write a post on scaling Transformers, implemented ideas from two NLP papers, and ... ended up with a super long post 😅 Since the topic is close to my heart, I decided to start a series of articles on it, focusing on one paper at a time. In all cases, let's dive in !

## Recommended reading

I expect you, dear reader, to be somewhat interested in AI if you're still here. NLP may not be your cup of tea though, and since following research trends requires time, you may also no be up to date. If that is the case, I would recommend you grab some coffee, and skim through these resources before going on:

- [Attention is all you're need](https://arxiv.org/pdf/1706.03762.pdf): this research paper introduced the Transformer model architecture, arguably changing the whole landscape of NLP with it, as LSTMs did back in the day
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): this post goes over the concepts presented in the above paper, it's a great resource to visualize how Transformer models work
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html): this post implements step by step the model described in the above paper, it's a great resource to understand hands-on how to translate the architecture into code
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf): this research paper introduced masked language modeling, a task which allows the pre-training of Transformer-based language models, usable (through subsequent fine-tuning) fo a wide range of downstream tasks

Okay, I hope that coffee was good ☕️ Now that you're up to date, let's _actually_ get to it 

## Transformers: "fat and slow"

As you may have realized by now, Transformer-based language models are currently dominating the NLP scene. They are the basis of SOTA models for most tasks, and their ability to scale depth-wise, contrary to recurrent models, have made them the de facto choice for any modern NLP pipeline. These models, unfortunately, come with major drawbacks: they are extremely fat and slow models compared to their predecessors. The basic BERT model for instance has upwards of 110 million trainable parameters - and that's considered small by 2020 standards 😱

Luckily, the pre-train / finetune framework introduced in the BERT paper somewhat alleviated this issue, as finetuning these models remains somewhat accessible to individuals and small organizations. It prompted [HuggingFace](https://huggingface.co) to create the now-famous [transformers](https://github.com/huggingface/transformers) library, which allows one to download and use most pre-trained models quite easily. However, the sheer size of the larger models still make them quite expensive to train and serve - and forget about pretraining these models from scratch, unless you're backed by a large organization. Due to this, independent researchers found themselves somewhat screwed, and started pushing back, e.g. the hilarious [Single Headed Attention RNN](https://arxiv.org/pdf/1911.11423.pdf) paper by Stephen Merity 😂

Recognizing the multiple issues with these larger than life models - financial, environmental, etc. - multiple solutions are emerging to solve them, among which:

- weight pruning, a well-known method to reduce model size
- weight quantization, a switch from 32-bit floating-point precision to 16-bit 
- knowledge distillation, a training method to create smaller "distilled" model using larger ones, through student-teacher training

While these methods do help in democratizing the usage of large models, none of them directly address the problem of training a large language model from nothing. Furthermore, multiple aspects of the Transformer architecture make it complicated to use on certain inputs, e.g. long texts as attention scales quadratically with input length.

To summarize, while the basic Transformer architecture can theoretically scale, in practice not so much 😔 So, what to do ? Well, innovations are needed to address the shortcomings of the architecture, and researchers from all around are now starting to deliver on this 🤩 It's unclear to me though how far off we are, so I've decided to progressively implement the papers I feel are going in the right direction in a [project](https://github.xom/r0mainK/outperformer), to see for myself. Each time I add a paper, I'll try to explain it in a post, like this one.

Okay, with this introduction done let's finally get down to business, and start this series with ...

## The Reformer

The first paper I'll cover is [Reformer: the Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf), coming from the Google Research lab. Acknowledging the limitations of the  architecture, the authors presented 3 ways to alleviate them, and the code they released actually contains a 4th innovation (I didn't realize this until reading the HuggingFace [post](https://huggingface.co/blog/reformer) on this paper). Said innovations are:

- Computing attention with an [LSH](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)-based approximation, reducing the complexity with regards to sequence length to log-linear time, instead of quadratic
- Chunking inputs fed to the feed-forward blocks
- Incorporating reversible residual layers for memory-efficient backpropagation
- Introducing axial positional encodings to scale these with long sequences

I won't be talking about the first because, although I was thrilled to see LSH being used in the context of Transformers, the paper I'll be covering in the next article introduces a better approximation of attention with regards to both quality and computation. I will also be skipping the last one because I intend to do a separate post to talk about embeddings at some point, in which I'll probably talk about this 🤗 So, as it's the easiest let's start with ...

### Chunking

In Transformer language models, each layer is divided in two: the attention block, and the feed-forward block. The latter is composed of two linear layers, which operate on the embedding dimension. Usually, the first linear layer maps the input to a much larger dimension, resulting in a large intermediate matrix, which is then mapped back to the embedding dimension after activation and dropout.

Noting that this intermediate matrix could be a memory bottleneck for large batches of lengthy sequences, the authors came up with a simple yet effective idea: to chunk the input along the sequence length dimension and apply the feed-forward block sequentially to each chunk. As the transformation is independent of all dimensions but the last, this does not change the output, and results in much smaller memory usage during the forward pass. Now the trade-off is of course time - but better slow things down than risk an OOM error. The implementation is pretty straightforward so I won't dwell too much (here `c` is the number of chunks in which the input is divided): 

```python
class ChunkedFeedForwardLayer(Module):
    def __init__(self, d_model, d_ff, dropout_rate, c):
        super(ChunkedFeedForwardLayer, self).__init__()
        self.linear_1 = Linear(d_model, d_ff)
        self.linear_2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout_rate)
        self.gelu = GELU()
        self.c = c

    def forward(self, x):
        chunks = x.chunk(self.c, dim=1)
        chunks = [self.linear_2(self.dropout(self.gelu(self.linear_1(chunk)))) for chunk in chunks]
        return torch.cat(chunks, dim=1)
```

### Reversible residual layers

{% katexmm %}
It may not be obvious, but when training a model a certain amount of tensors are cached during the forward pass, for backpropagation purposes. Specifically, if $y=f(x)$ then $x$, what is called the activation, needs to be cached. In Pytorch this happens under the hood, and can be disabled by encapsulating computations with the `torch.no_grad` function. At first glance you might think "big deal", but for deep models storing activations at each layer induces an important memory cost. This is where reversibility comes into play: if each layer's activation could be computed using only its output, then since backpropagation is done in reverse order there would be no need to store activations, removing the problem entirely, and resulting in large memory gains. By leveraging residual connections, the authors of [this](https://arxiv.org/pdf/1707.04585.pdf) Computer Vision paper managed to do exactly that 😎 The authors of the Reformer then applied it to Transformers. Let's see how they did it.
{% endkatexmm %}

#### **Some math**

{% katexmm %}
The way a residual connection works is by adding the input to a given layer to its output. In Transformer models, these connections exist between both blocks of each layer. Naming these blocks $F$ and $G$, the output of a given layer is:
{% endkatexmm %}

{% katex display %}
y = G(F(x) + x) + F(x)
{% endkatex %}

Now, this layer as is isn't reversible, however it can be so with a slight change in perspective, namely by using _pairs_ of inputs and output instead:

{% katex display %}
y_1 = x_1 + F(x_2)\newline
y_2 = x_2 + G(y_1)
{% endkatex %}

And that's it, with this the layer is now reversible:

{% katex display %}
x_2 = G(y_1) - y_2\newline
x_1 = F(x_2) - y_1
{% endkatex %}

{% katexmm %}
So, how do we get to two inputs and back ? Given the stack of reversible layers, the input is first duplicated along a new axis: $x = [x^T,x^T]^T$ and then fed to the stack. After propagation through the whole stack, the output is simply averaged along the newly created axis. Now you may be wondering whether this changes anything 🤔 And you'd be right, this does indeed slightly change the architecture. For instance, for a one layer stack:
{% endkatexmm %}
{% katex display %}
y = \frac{y_1 + y_2}{2}
= \frac{2x +  F(x) + G(F(x) + x)}{2}
{% endkatex %}

If you forget about the division, you should see that this amounts to adding a residual connection between the start and end of each layer. While this does make the model slightly different, it's not a groundbreaking change, and shouldn't affect performances - adding residual connections is generally seen as an improvement until proven otherwise.

Now that the math is out of the way, let's move on to the implementation !

#### **Extending `torch.autograd` for layer-wise backpropagation**

First, let's see how to modify backpropagation. The [documentation](https://pytorch.org/docs/stable/notes/extending.html) here is quite helpful: the `torch.autograd.Function` class need to be extended, by implementing two static methods (one for each pass):

```python
class ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, stack):
        for layer in stack:
            x = layer(x)
        ctx.y = x.detach()
        ctx.stack = stack
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        for layer in ctx.stack[::-1]:
            y, dy = layer.backward(y, dy)
        return dy, None
```

In the `forward` method, the input is propagated through each layer of the stack, while in the `backward` method the reverse order is used. The `ctx` variable allows us to store anything needed for backpropagation (the output gradient is provided) - here the final output is stored in `ctx.y` and the stack of layers is stored in `ctx.stack`, as both will be needed. Regarding the `None` returned in the backward pass, it has to be provided because one of the layers has an optional key argument. While this does not affect the gradient computation, Pytorch still expects two outputs, but unused ones can be defaults to `None`.

Now, let's define the stack of layers:

```python
class ReversibleStack(Module):
    def __init__(self, stack):
        super(ReversibleStack, self).__init__()
        self.layers = ModuleList([ReversibleLayer(layer_1, layer_2) for layer_1, layer_2 in stack])

    def forward(self, x):
        out = ReversibleFunction.apply(x.expand(2, *x.shape), self.layers)
        return out.mean(dim=0)
```

Nothing too fancy here, the stack is created via the list of paired blocks, by combining them in a `ReversibleLayer`. In the `forward` method the output is yielded by the `ReversibleFunction.apply` method, ensuring it will be used during backpropagation. As I mentioned earlier, inputs are duplicated along a new axis, and the output is averaged along that axis. 

{% katexmm %}
Now before implementing the `ReversibleLayer` let's step back for a moment, as I've omitted a rather important detail. You may have noted from the math above that to reconstruct activations $x_1$ and $x_2$ in the backward pass, $F(x_2)$ and $G(y_1)$ will need to be recomputed - unlike the output they are not directly available. This assumes that $F$ and $G$ are deterministic functions - _which they are not_.
{% endkatexmm %}
Indeed, linear layers used in both blocks are followed by dropout, which randomly nullifies some of the outputs. This is problematic, as it means the reconstruction will fail if things are left as is - and removing dropout would hurt performances, so that's a no-go. Luckily this isn't the situation setting in which non-determinism would be a problem, and the solution to it already exists 🤩

If you think about it, the same situation is created if training is put on hold, i.e. checkpointed. Citing the Pytorch [documentation](https://pytorch.org/docs/stable/checkpoint.html) on the subject: 

> Checkpointing is implemented by rerunning a forward-pass segment for each checkpointed segment during backward. This can cause persistent states like the RNG state to be advanced than they would without checkpointing. By default, checkpointing includes logic to juggle the RNG state such that checkpointed passes making use of RNG (through dropout for example) have deterministic output as compared to non-checkpointed passes.

The logic they are referencing can be found [here](https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html), in the `CheckpointFunction`. However, integration to our `ReversibleFunction` can't be done, since the order in which layers are applied will be, well, reversed. To circumvent  this, let's simply create a container layer in which RNG states for the CPU and GPU(s) can be preserved and reset:

```python
class DeterministicLayer(Module):
    def __init__(self, layer):
        super(DeterministicLayer, self).__init__()
        self.layer = layer
        self.cpu_state = None
        self.gpu_devices = None
        self.gpu_states = None

    def forward(self, x, backward=False):
        if self.training:
            self.cpu_state = torch.get_rng_state()
            self.gpu_devices, self.gpu_states = get_device_states(x)
        if backward:
            torch.set_rng_state(self.cpu_state)
            with fork_rng(devices=self.gpu_devices, enabled=True):
                set_device_states(self.gpu_devices, self.gpu_states)
                return self.layer(x)
        return self.layer(x)
```

You'll notice this is the layer with a key argument 🧐 Anyway, now that this detail is covered, let's finish this off ! 

#### **Implementing the reversible layer, and _more math_**

Up until now, implementing this paper has been relatively straightforward.

```python
class ReversibleLayer(Module):
    def __init__(self, layer_1, layer_2):
        super(ReversibleLayer, self).__init__()
        self.layer_1 = DeterministicLayer(layer_1)
        self.layer_2 = DeterministicLayer(layer_2)

    def forward(self, x):
        x1, x2 = (t.squeeze() for t in x.chunk(2))
        with torch.no_grad():
            y1 = x1 + self.layer_1(x2)
            y2 = x2 + self.layer_2(y1)
        return torch.stack([y1, y2])

    def backward(self, y, dy):
        y1, y2 = (t.squeeze() for t in y.chunk(2))
        dy1, dy2 = (t.squeeze() for t in dy.chunk(2))
        y1.requires_grad = True
        y2.requires_grad = True
        with torch.enable_grad():
            y2_no_res = self.layer_2(y1, backward=True)
            y2_no_res.backward(dy2, retain_graph=True)
        with torch.no_grad():
            x2 = y2 - y2_no_res
            dx1 = dy1 + y1.grad
            y1.grad = None
        x2.requires_grad = True
        with torch.enable_grad():
            y1_no_res = self.layer_1(x2, backward=True)
            y1_no_res.backward(dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - y1_no_res
            dx2 = dy2 + x2.grad
            x2.grad = None
        x = torch.stack([x1, x2])
        dx = torch.stack([dx1, dx2])
        del y, y1, y2, dy, dy1, dy2, y1_no_res, y2_no_res
        return x, dx
```

{% katexmm %}

_Not anymore 😱_

As a Pytorch user, you may not be used (anymore) to implementing the backward pass of a `Module`. Although most of the work is still going to be done under the hood, some explanations are required for the input gradients computation. I left this out until now to finish with a bang 😈 In all cases, let's quickly unpack the rest first:

- inputs and outputs are unpacked and repacked at the start and end of each pass
- both blocks are wrapped in the `DeterministicLayer`, ensuring dropout is applied the same way forward and back during training
- variables are explicitly deleted at the end of the backward pass, to free memory faster than with garbage collection
- the forward and backward pass compute $x_1$, $x_2$, $y_1$ and $y_2$ exactly as I described, without storing activations 

If you go back to the code, it should be more clear. Now, what's up with the gradients of $x_1$ and $x_2$ 🤨 ? To understand, it helps to go back to basics: the gradient of a node in the computation graph reflects changes that happen downstream.

In the case of $x_1$, a small change in value will linearly modify $y_1$, but since $y_2$ is computed through $G(y_1)$ its gradient with respect to $dy_2$ must also be added. There is no need to go further in the computation graph since the backpropagation is done layer-wise, meaning all other downstream effects are taken into account in $dy_1$ and $dy_2$. In all cases, this translates to:
{% endkatexmm %}

{% katex display %}
dx_1 = dy_1 + \frac{\delta G}{\delta y_1}dy_2
{% endkatex %}

{% katexmm %}
The first component of this sum is an input of the backward pass, and the second is stored in $y_1$ after backpropagation on $G(y_1)$, which is done manually with `y2_no_res.backward(dy2, retain_graph=True)` This leads to the first gradient: `dx1 = dy1 + y1.grad`

Let's apply the same method to derive the gradient of $x_2$. As before, a small change in value will affect linearly $y_2$, and since $y_1$ is computed through $F(x_2)$ its gradient with respect to $dy_1$ must also be taken into account. Unlike before however, the computation can't stop here and must be taken a step further, since this small change in $y_1$ also affects $y_2$. All of this translates to the following:
{% endkatexmm %}

{% katex display %}
dx_2 = dy_2 + \frac{\delta F}{\delta x_2}dy_1 + \frac{\delta F}{\delta x_2}\frac{\delta G}{\delta y_1}dy_2 = dy_2 + \frac{\delta F}{\delta x_2}dx_1
{% endkatex %}

{% katexmm %}
Now that the gradient is explicit, the computation can be done the same way. The second term is created after manually backpropagating on $F(x_2)$ with `y1_no_res.backward(dx1, retain_graph=True)`, and is stored in $x_2$, which leads to the second gradient: `dx2 = dy2 + x2.grad`.
{% endkatexmm %}

Anyway, that's what's up with the gradients ! Now, I know that I didn't explain the math in the most rigorous of ways - I've been out of school for a while and it shows when it comes to this 😅 If you want a better explanation I recommend you go ahead and check the paper that introduced Reversible Layers - drawing the computation graph of a single layer will also help in the process.

## In conclusion

And that wraps it up for the first part of the series ! I hope you enjoyed this post, I know I had a lot of fun researching and writing it 😁 If you want to see how it all comes together in an actual model, you can check it out on GitHub [here](https://github.com/r0mainK/outperformer), and if you have any feedback (good or bad), I'll be glad to hear it, so don't hold back 🙏

As I mentioned in the introduction, I'll be releasing the second part of the series in a short while - the code is already pushed on GitHub, and I'm halfway through the writing process. I'd like to keep this series going, but haven't found the paper I want to implement yet. Do you know any ? If you've got any ideas (even self-promotion) hit me up !

Finally, I'd like to thank everybody that contributed to the resources below - you should check them out if you've got the time.

## Resources

### Code

- [Official Reformer code](https://github.com/google/trax/blob/master/trax/models/reformer): the code released by the authors of the Reformer paper, in Jax, which I didn't end up using as much as I don't know Jax 😅

- [RevTorch package](https://github.com/RobinBruegger/RevTorch): this was my main resource for understanding and implementing the Reverse Residual layers, along with the paper 👌

- [performer-pytorch package](https://github.com/lucidrains/performer-pytorch): a more complete library implementing both the Reformer and the next paper I'll talk about, the Performer. As the author tried to stay as close as possible to the Jax code and contributed to the above repo, it was a useful resource to check I wasn't forgetting anything (e.g. deterministic dropout) 😁

### Papers

- [Reformer: the Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf), by Nikita Kitaev, Łukasz Kaiser and Anselm Levskaya

- [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf), by Aidan N. Gomez, Mengye Ren, Raquel Urtasun and Roger B. Grosse
