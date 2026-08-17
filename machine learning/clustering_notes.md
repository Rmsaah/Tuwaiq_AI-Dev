# Clustering notes — `spotify-cluster.ipynb`

---

## 1. The problem with K-Means

You have to tell it `k` before it runs. Nothing in the data says what `k` is.

Two tools help you pick. **Elbow** and **silhouette**.

---

## 2. Inertia

**Add up the distance from every point to its own cluster's centre, squared.**

```
        cluster 0                     cluster 1

     o                                     o
      \                                   /
   o -- C                              C ---- o
      /                                 \
     o                                   o

   inertia = sum of every arrow, squared
```

Also written **WCSS**, within-cluster sum of squares. In sklearn it's `kmeans.inertia_`.
Low inertia means tight clusters.

### Worked example

Six points on a line: `1, 2, 3, 10, 11, 12`

**One cluster.** Centre = mean = 6.5

```
(1-6.5)² + (2-6.5)² + (3-6.5)² + (10-6.5)² + (11-6.5)² + (12-6.5)²
  30.25  +   20.25  +   12.25  +   12.25   +   20.25   +   30.25   = 125.5
```

**Two clusters, split well.** `1 2 3` centre 2, `10 11 12` centre 11

```
(1+0+1) + (1+0+1) = 4
```

**Two clusters, split badly.** `1 2 3 10` centre 4, `11 12` centre 11.5

```
(9+4+1+36) + (0.25+0.25) = 50.5
```

`4` beats `50.5`. **That is how K-Means decides.** Inertia isn't a report card printed at the
end, it's the score the algorithm plays for.

### It is the algorithm's whole job

```
1. drop k centres at random
2. each point joins its nearest centre           <- inertia drops
3. each centre moves to the mean of its points   <- inertia drops
4. repeat 2 and 3 until nothing moves
```

Every step can only lower it. It stops when it can't lower it any further.

That's what `n_init=10` is for in our code. Step 1 is random, so a run can get stuck in a bad
layout. It runs the whole thing 10 times from 10 random starts and keeps the **lowest inertia**.

### Two things to know

**Why squared?** It punishes far-away points hard, and it makes the mean the mathematically best
centre. That's why step 3 is "move to the mean" and nothing more complicated.

**The number itself is meaningless.** Ours is 153,804, in squared units of scaled features. It
only means something next to another inertia on the same data. Never quote it alone.

### What inertia cannot tell you

It measures **tightness inside** a cluster. It never looks at the **gap between** clusters.

```
    (o o o)(x x x)              (o o o)     (x x x)
     touching                    clear gap

    inertia: identical           inertia: identical
    silhouette: low              silhouette: high
```

That's why you need both. And it's why inertia always shrinks as k grows, right down to 0 when
every point is its own cluster. So you read the **drop**, not the value.

---

## 3. Elbow method

### Why you can't just minimise it

More clusters always shrinks it. Give every song its own cluster and inertia hits 0.

```
inertia
  |*
  | *
  |  *
  |   *
  |    * * * * * * *  <- keeps falling forever, meaningless
  +------------------
    k ->
```

### The actual trick

Don't read the value. Read **how much it drops** each time you add a cluster.

**Drop** = the inertia at this k, subtracted from the inertia at the k before it.

```
k=5  ->  165,108
k=6  ->  153,804          drop = 165,108 - 153,804 = 11,304
```

That 11,304 is what the sixth cluster *bought you*. It's the only thing worth reading.

| Adding a cluster... | Drop |
|---|---|
| splits a real group in two | **big** |
| splits a group that was already fine | small |

The bend is where the big drops stop. That's the elbow.

### Tiny example

Six points on a line: `1, 2, 3, 10, 11, 12`

| k | split | inertia | drop |
|---|---|---|---|
| 1 | all together | 125.5 | — |
| 2 | `1 2 3` / `10 11 12` | 4.0 | **121.5** |
| 3 | `1 2 3` / `10 11` / `12` | 2.5 | 1.5 |
| 4 | `1` / `2 3` / `10 11` / `12` | 1.0 | 1.5 |

k=2 removed the real gap. Everything after that is just chopping. **Elbow = 2.**

### Our curve

| k | inertia | drop |
|---|---|---|
| 2 | 224,497 | — |
| 3 | 199,598 | 24,899 |
| 4 | 180,937 | 18,661 |
| 5 | 165,108 | 15,829 |
| **6** | **153,804** | **11,304** |
| 7 | 146,031 | 7,773 |
| 8 | 139,991 | 6,040 |
| 9 | 134,577 | 5,414 |
| 10 | 129,396 | 5,181 |

```
drop in inertia each time you add one more cluster

  k=3 |████████████████████████   24,899
  k=4 |███████████████████        18,661
  k=5 |████████████████           15,829
  k=6 |███████████                11,304   <- last big one
      |- - - - - - - - - - - - - - - - - -  the cliff
  k=7 |████████                    7,773
  k=8 |██████                      6,040
  k=9 |█████                       5,414
 k=10 |█████                       5,181
```

The drop is cut in half between k=6 and k=7. After that every extra cluster buys the same tiny
amount, which is the flat part of the curve. **Elbow is around 5–6.**

---

## 4. Silhouette score

Elbow is a judgement call. Silhouette gives you an actual number.

### What it measures, per point

```
       its own cluster              nearest other cluster
    .------------------.          .------------------.
    |    o    o        |          |      o     o     |
    |      \           |          |    o    o        |
    |   a   X ---------|-- b -----|--> o             |
    |      /           |          |         o        |
    |    o             |          |                  |
    '------------------'          '------------------'

    a = average distance from X to its own cluster mates
    b = average distance from X to the nearest other cluster
```

```
score = (b - a) / whichever of a, b is bigger
```

Plain words: **is X closer to its own group than to the next group?**

### Reading the number

| Score | a vs b | Meaning |
|---|---|---|
| near **+1** | b much bigger | deep inside its own cluster |
| near **0** | a = b | sitting on the border |
| below **0** | a bigger | closer to a *different* cluster — misplaced |

### Tiny example

| a | b | Calculation | Score |
|---|---|---|---|
| 2 | 8 | (8 − 2) / 8 | **+0.75** — good |
| 5 | 5 | (5 − 5) / 5 | **0.00** — on the fence |
| 8 | 2 | (2 − 8) / 8 | **−0.75** — wrong cluster |

The score you report is the **average over every point**.

### The score answers two different questions

Don't mix them up. One number, two readings.

| Reading | Question | Our answer |
|---|---|---|
| **where the peak is** | which k should I pick? | k = 6 |
| **how high the peak is** | are these clusters any good? | 0.155, so barely |

The peak tells you nothing about quality. It only says 6 beat every other k we tried.

### How high is high

| Score | What it means |
|---|---|
| 0.7+ | clear gaps between groups |
| 0.5 | reasonable structure |
| 0.25 | weak |
| **0.155** | **ours — groups touch each other everywhere** |
| 0 | no structure at all |
| below 0 | points are in the wrong clusters |

At 0.155 the average song sits almost exactly as close to a neighbouring cluster as to its own.
Section 6 covers why that was always going to happen here.

### Our curve

| k | silhouette |
|---|---|
| 2 | 0.1334 |
| 3 | 0.1414 |
| 4 | 0.1501 |
| 5 | 0.1507 |
| **6** | **0.1547** ← peak |
| 7 | 0.1415 |
| 8 | 0.1393 |
| 9 | 0.1461 |
| 10 | 0.1387 |

---

## 5. Reading both together

Part 2 of the notebook draws them side by side. One `KMeans` fit per k gives both numbers, so
they come out of the same loop.

### The left plot — elbow

```
inertia
224k  *
        \
200k      *
            \
181k          *
                \
165k              *
154k                 *___
146k                      *___
140k                           *___
135k                                *___
129k                                     *
      2    3    4    5    6    7    8    9   10
                       k
```

Read the **shape**, not the height.

| Region | k | Average drop per extra cluster |
|---|---|---|
| steep | 2 to 5 | about **19,800** each |
| flat | 6 to 10 | about **6,100** each |

*(steep: 224,497 - 165,108 = 59,389 over 3 steps. flat: 153,804 - 129,396 = 24,408 over 4 steps.)*

It bends at **k = 5 or 6** and flattens after. Past that, a new cluster buys a third of what the
early ones did.

**But it's a soft bend, not a corner:**

```
   a strong elbow                   ours

   *                                *
    \                                 \
     \                                  *
      *                                    *
      |                                       *
      *--*--*--*--*                              *--*--*--*
        sharp corner,                     gentle curve,
        k is obvious                      k is a judgement call
```

No sharp elbow means **there is no natural number of groups**. Same message as the low
silhouette. Not a failure, a finding.

### The right plot — silhouette

```
0.155 |                        *   <- peak, k=6
0.150 |            *     *
0.146 |                                    *
0.141 |      *                 *
0.139 |                              *           *
0.133 |*
      2     3     4     5     6     7     8     9    10
                          k
```

Read the **peak**. Climbs 2 to 6, tops out at **0.1547**, then falls off a cliff to 0.1415 at k=7.

That cliff is the useful bit. It means splitting into 7 starts **cutting through real groups**
instead of separating them.

### Three traps in these plots

**1. The right y-axis is lying to you.** It runs 0.133 to 0.155, a range of **0.022**. Matplotlib
zooms all the way in, so a tiny difference looks like a mountain.

```
   what you see                 what it is on a real 0 to 1 scale

   0.155 |    /\                1.0 |
         |   /  \                   |
         |  /    \                  |
   0.133 |_/      \___          0.0 |_*_*_*_*_*_*_*_*_*_  (all squashed here)
```

Every k scores badly. k=6 is just the least bad.

**2. The bump at k=9.** It climbs back to 0.1461. Ignore it. That's noise from
`sample_size=5000` wobbling inside a 0.02 band. Don't build a story on it.

**3. Neither plot proves clusters exist.** They only answer *"if I must pick a k, which one?"*
Whether the clusters are worth anything is settled in Part 5, by whether they can be named.

### Conclusion

| Tool | Says |
|---|---|
| Elbow | 5 or 6 |
| Silhouette | 6 |

**k = 6.** They agree, so we don't have to argue about it. That agreement is the whole reason you
run both. If the elbow said 4 and silhouette said 8, you'd have nothing.

---

## 6. Why 0.15 is low, and why we keep going

```
   silhouette 0.70                 silhouette 0.15  (ours)

    ooo          xxx                    o x o o x
   ooooo        xxxxx                  x o o x o x
    ooo          xxx                    o x o x o
                                         x o o x

   real gap between groups           one cloud, sliced up
```

Songs don't come in separated islands. Audio features slide into each other continuously — there is no empty space between "danceable" and "very danceable."

So K-Means isn't *finding* gaps here. It's **cutting a cloud into useful regions**. Low score, still useful — and section 10 shows the regions are clean and nameable.

Don't oversell it in the presentation. Say the number and say what it means.

---

## 7. PCA — how we drew the picture

### The short version: PCA is a camera

You have a chair. A chair is a 3D thing.

You want to send a picture of it to a friend. A picture is flat. So you have to squash 3D into 2D.

**Where do you stand?**

```
   photo from directly above          photo from the side

         ________                            ___
        |        |                          |   |
        |        |                          |   |
        |________|                          |___|___
                                            |       |
                                            ||     ||
                                            ||     ||

      a square. useless.               oh, it's a chair.
```

Same chair. Same flat picture. **One angle is much better than the other.**

**PCA walks around the object and finds the good angle.**

### What "good angle" means

The angle where things look the **most spread out**.

```
   bad angle                     good angle

   all bunched up                 spread out
      o o                        o
      oo                              o
      o                    o                   o
                                   o
   can't tell them apart        can tell them apart
```

If everything piles into one dot, you learn nothing.

### What ours was

Our "chair" is 28,351 songs, each described by **9 numbers**. That's a 9-dimensional shape.
Nobody can see 9 dimensions.

So we asked PCA for the best flat photo. That photo is the scatter plot in Part 4.

### The catch

A chair photo loses a little. A 9-dimension photo loses a lot. Ours kept **36%**, threw away
**64%**.

```
   in the photo                    in real life

   two people look like            they're 10 metres apart,
   they're touching                one is just behind the other

     (o)(o)                          (o) . . . . . . . (o)
```

Two clusters can look squashed together in our plot and be far apart in the real 9 dimensions.
**The plot is not the proof. The silhouette score is.**

### The most important part

**PCA did not make the groups.**

```
   1. we grouped the songs   <- using all 9 numbers
   2. THEN we took a photo   <- PCA
```

Like sorting your toys into boxes first, and *then* photographing the boxes. The picture didn't
decide which toy goes where.

---

### Worked example: 2 features down to 1

Six songs, already scaled, so they're centred on 0.

| song | energy | loudness |
|---|---|---|
| A | -1.5 | -1.4 |
| B | -1.0 | -1.1 |
| C | -0.3 | -0.2 |
| D | +0.4 | +0.5 |
| E | +1.0 | +1.1 |
| F | +1.4 | +1.1 |

**Step 1 — plot it**

```
 loudness
    +1.5 |                           E  F
         |                      D
     0.0 |                 C
         |
    -1.5 |   A  B
         +------------------------------
          -1.5                      +1.5
                    energy
```

They sit on a diagonal. Two columns, but really one piece of information — loud songs are
energetic songs.

**Step 2 — find the direction with the most spread**

```
 loudness
    |                     E F
    |                D  /
    |           C    /          <- this diagonal is where
    |             /                the points are spread out
    |     A  B /
    +------------------- energy
              ^
         call it pc1

    perpendicular to it, the points barely move
                    ^
               call it pc2
```

**Step 3 — rotate so pc1 is the new x-axis**

```
   pc2
    |
  --A-----B---------C------D------E---F--->  pc1
    |
       everything lives on this line now, pc2 is flat
```

**Step 4 — the recipe**

pc1 and pc2 are mixtures of the old two. PCA works out the weights:

```
pc1 =  0.724 * energy + 0.690 * loudness
pc2 = -0.690 * energy + 0.724 * loudness
```

Song A: `0.724 x (-1.5) + 0.690 x (-1.4)` = **-2.05**

| song | pc1 | pc2 |
|---|---|---|
| A | **-2.05** | 0.02 |
| B | **-1.48** | -0.11 |
| C | **-0.36** | 0.06 |
| D | **+0.63** | 0.09 |
| E | **+1.48** | 0.11 |
| F | **+1.77** | -0.17 |

pc1 runs from -2.05 to +1.77. pc2 never leaves plus or minus 0.17.

**Step 5 — drop pc2**

| | spread it holds |
|---|---|
| pc1 | **99.49%** |
| pc2 | 0.51% |

Two columns became one, and 0.5% was lost.

### In our notebook: 9 down to 2

```
danceability
energy
speechiness
acousticness          PCA                pc1
instrumentalness   --------->            pc2
liveness
valence
tempo
duration_min
```

This time the 9 features **don't** all say the same thing. Danceability isn't tempo, acousticness
isn't liveness. There's no single diagonal to collapse onto.

| | spread it holds |
|---|---|
| pc1 | 18.8% |
| pc2 | 16.8% |
| **kept in the plot** | **35.6%** |
| **thrown away** | **64.4%** |

### Why the axes mean what they mean

We didn't choose the labels on the plot. **PCA picked two directions purely by "where is the
data most spread out", and then we read what they turned out to be.**

`pca.components_` gives the recipe — how much each original feature counts toward each axis.

| Feature | pc1 | pc2 |
|---|---|---|
| danceability | -0.23 | **+0.56** |
| energy | **+0.63** | +0.22 |
| speechiness | -0.11 | +0.34 |
| acousticness | **-0.59** | -0.20 |
| instrumentalness | +0.06 | -0.29 |
| liveness | +0.26 | -0.03 |
| valence | -0.01 | **+0.59** |
| tempo | +0.30 | -0.11 |
| duration_min | +0.15 | -0.19 |

**Reading it:** big number = that feature drives the axis. The sign says which end.

- **pc1** — energy **+0.63** pulls right, acousticness **-0.59** pulls left. Two big numbers
  pulling opposite ways, so pc1 is an axis running `acoustic <---> energetic`.
- **pc2** — valence **+0.59** and danceability **+0.56**, both pulling up. So pc2 runs
  `downbeat <---> upbeat`.

Those names are **ours**, not sklearn's. sklearn returns numbers; naming the axis is
interpretation, the same job as naming a cluster.

And keep it loose. pc1 only holds 18.8% of the spread, so "energetic vs acoustic" is the theme of
the axis, not a definition of it.

### Why PCA landed on those two patterns

PCA didn't pick features. **It looked for things that always move together, and took the two
biggest patterns it found.**

**Step 1 — spot the hidden slider.** Four songs:

```
            energy    acousticness
   song A     0.9         0.02
   song B     0.8         0.05
   song C     0.3         0.70
   song D     0.2         0.85
```

Energy up, acousticness down. Every time. These aren't two facts, they're **one fact told
twice** — one slider wearing two hats:

```
   acoustic  <--------------------->  energetic
```

**Step 2 — rank the sliders, biggest first.** Every strong pair in our 9 features:

```
   acousticness  <-> energy          -0.546   <- biggest thing in the data
   danceability  <-> valence         +0.334   <- second biggest
   tempo         <-> danceability    -0.185
   danceability  <-> speechiness     +0.183
   instrumental. <-> valence         -0.174
   liveness      <-> energy          +0.164
   tempo         <-> energy          +0.151
   energy        <-> valence         +0.150
```

The biggest becomes **pc1**. Opposite signs because they move opposite ways:

```
   energy        +0.63     <- one end of the seesaw
   acousticness  -0.59     <- the other end
```

**Step 3 — pc2 must stand at a right angle.** pc1 used up the energy/acoustic slider, so pc2
isn't allowed to overlap with it.

```
       pc2
        |
  ------+------ pc1
        |
```

It goes hunting for the biggest *leftover* pattern, which is `danceability <-> valence, +0.334`.
Both move the same way, so both get a plus:

```
   valence       +0.59
   danceability  +0.56
```

**Step 4 — the small numbers are the supporting cast.** Everything else joins whichever axis it
already leans toward:

| Feature | Lands on | Because |
|---|---|---|
| tempo **+0.30** | pc1 | tempo/energy +0.15, tempo/acoustic -0.11, same direction as energy |
| liveness **+0.26** | pc1 | liveness/energy +0.16 |
| speechiness **+0.34** | pc2 | speechiness/danceability +0.18 |
| instrumentalness **-0.29** | pc2 | instrumentalness/valence -0.17, so it points down |

In one line:

```
   pc1  =  the -0.546 pattern    (energy vs acousticness)
   pc2  =  the +0.334 pattern    (valence with danceability)
```

### What "variance kept: 35.6%" means

Part 4 prints this:

```
variance kept: [0.188 0.168] -> 35.6 %
```

**Variance is just "how spread out the data is".** PCA rewrites the 9 features as 9 new columns,
sorted so the most spread-out one comes first. We kept the first two.

| | Holds |
|---|---|
| pc1 | 18.8% of the total spread |
| pc2 | 16.8% |
| **pc1 + pc2** | **35.6%** — what the plot shows |
| pc3 to pc9 | 64.4% — thrown away |

Think of a 9-page story rewritten so the most important part is page 1. We only printed pages 1
and 2. Those two pages carry about a third of the story.

Here's all nine:

```
   pc1   18.8%  ##################
   pc2   16.8%  ################
   pc3   12.6%  ############
   pc4   11.0%  ###########
   pc5   10.9%  ###########
   pc6   10.1%  ##########
   pc7    9.2%  #########
   pc8    6.5%  ######
   pc9    4.2%  ####
```

Nearly flat. If the 9 features were completely unrelated, each would hold exactly **11.1%**. Ours
run 18.8 down to 4.2, so there *is* some structure, but barely.

Compare a dataset where everything is tightly linked — pc1 alone would be 70% or 80%, and a 2D
plot would show you almost everything.

**So 35.6% is the warning label on the scatter plot.** Not a mistake, just an honest measure of
how much a flat picture of 9 dimensions can show. It's the same story the 0.155 silhouette tells:
nine features, nine mostly separate things, no big pattern to find.

### Where PCA sits

| | |
|---|---|
| Clustering happened in | 9 dimensions |
| PCA runs | afterwards, on the same data |
| Does PCA change the clusters | **no** — labels were already assigned |
| PCA is for | drawing the scatter plot |

`pc1` and `pc2` are not real features, and the plot is a shadow missing most of the detail. Two
clusters that look smeared together there may be far apart in the real 9 dimensions.

**PCA is the camera. Silhouette is the evidence.**

---

## 8. One practical note

Silhouette compares every point to every other point. 28,351 songs = **800 million distances** per call, ×9 values of k.

```python
silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
```

Measures 5,000 random points instead. Same answer, seconds instead of minutes.

---

## 9. What each cell does

| Section | Does | Slide 21 step |
|---|---|---|
| Setup | imports, load csv | choose dataset |
| Part 1 | drop 5 empty names, drop the 4-second track, dedup to 28,351 songs | preprocess |
| Part 1 · Feature selection | pick 9 audio features, show their ranges | choose dataset |
| Part 1 · Scaling | `StandardScaler`, confirm mean 0 std 1 | preprocess |
| Part 2 | loop k=2..10, collect inertia + silhouette, plot both | evaluate |
| Part 3 | fit K-Means and GMM at k=6, score both | fit two algorithms |
| Part 4 | PCA to 2D, scatter both algorithms side by side | visual inspection |
| Part 5 | cluster means raw, then as a heatmap in std devs | interpret |
| Part 5 · Size and popularity | how big each cluster is, mean popularity | interpret |
| Part 5 · Clusters against genre | crosstab, cluster vs playlist_genre | interpret |
| Part 5 · Naming the clusters | names + top songs in each | practical meaning |
| Part 6 | K-Means vs GMM table | present |

### Why these 9 features

In, because they describe how a song **sounds**:
`danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_min`

Out, and why:

| Dropped | Reason |
|---|---|
| `loudness` | +0.68 with `energy`. Same thing twice, counted twice in the distance. |
| `key`, `mode` | categories. Distance between "key 3" and "key 7" means nothing. |
| `track_popularity` | an outcome, not a sound. We profile it *after*, per cluster. |
| `playlist_genre` | a label. Feeding it in would just hand back the genres. |

### Why scaling is mandatory

| Feature | Range |
|---|---|
| tempo | 35 → 239 |
| duration_min | 0.5 → 8.6 |
| danceability | 0.08 → 0.98 |

Unscaled, a 40 BPM tempo gap outweighs *every* 0–1 feature combined. Tempo would decide all six clusters on its own.

---

## 10. What we found

**k = 6.** Each cluster is driven by one feature standing out.

| # | Name | Songs | Defined by | Top genre | Mean popularity |
|---|---|---|---|---|---|
| 0 | upbeat and danceable | 9,270 | valence **+0.74**, danceability +0.58 | latin 24% | 41.2 |
| 1 | fast, hard, low mood | 7,307 | danceability **−0.77**, valence −0.60, tempo +0.44 | rock 28% | 38.6 |
| 2 | instrumental | 2,330 | instrumentalness **+2.82** | **edm 56%** | 29.7 |
| 3 | acoustic and calm | 3,595 | acousticness **+1.87**, energy −1.48 | **r&b 33%** | **42.7** |
| 4 | live recordings | 1,951 | liveness **+2.69** | edm 25% | 36.1 |
| 5 | spoken word and rap | 3,898 | speechiness **+1.99** | **rap 56%** | 40.6 |

*(numbers are standard deviations from the average song)*

### Did the clusters find the genres?

We never showed the model a genre. Then we checked:

```
cluster 5  ->  56% rap
cluster 2  ->  56% edm
cluster 3  ->  33% r&b
cluster 0  ->  24% latin  (most mixed)
```

**Partly.** Rap and edm come out clearly because they have a sound signature — rap is speech, edm is instrumental. Pop, latin and rock smear across every cluster.

That is the honest headline: **clusters are built from sound, genres are built from playlists.** The 1,686 songs carrying more than one genre label were already the warning sign in the EDA.

---

## 11. K-Means vs Gaussian Mixture

### What GMM is

**Gaussian Mixture Model.** It assumes the data is a pile of overlapping bell curves, and works
out which bells they are.

```
   K-Means                          GMM
   "which centre is nearest?"       "which bell most likely made this point?"

   ( o o o )( x x x )               ( o o o( x o )x x )
                                          ^^^^^^
   hard border                      overlap allowed
```

K-Means says *"song 412 is in cluster 3."*
GMM says *"song 412 is 55% cluster 3, 40% cluster 1, 5% elsewhere."*

### Why we ran it alongside

Slide 21 asks for "K-Means + one other method". But GMM is also the *right* second opinion,
because it relaxes the one assumption K-Means can't escape:

```
K-Means assumes:  clusters are round and equal-sized
GMM asks:         what if they're not?
```

If our clusters were long stretched shapes, GMM would win and we'd know K-Means was the wrong
tool. That makes it a real test, not a formality.

It also runs at the same k=6, on the same scaled data, so the two are directly comparable.

### The result

| | K-Means | Gaussian Mixture |
|---|---|---|
| Clusters | 6 | 6 |
| **Silhouette** | **0.155** | **−0.013** |
| Smallest / largest | 1,951 / 9,270 | 2,244 / 10,431 |
| Cluster shape | round, equal-ish | stretched ellipses, any size |
| Assignment | one cluster, hard | probability per cluster |
| Profiles | one feature stands out per cluster | all six look alike |

### Why GMM scored negative

```
   K-Means                          GMM
   round, side by side              ellipses that overlap

    ( o o )( x x )                  ( o o( x o )x x )
                                          ^^^^^^
                                    points here sit inside
                                    two clusters at once
```

Silhouette asks *"is this point closer to its own group?"* GMM lets clusters overlap, so for many points the answer is no. Negative score.

**GMM isn't broken** — it optimises likelihood, not compactness. It's the wrong tool when the metric and the goal are both about separation.

---

## 12. Recommendation

**Use K-Means, k = 6.**

- Higher silhouette (0.155 vs −0.013).
- Every cluster has one defining feature, so every cluster gets a name.
- GMM's six clusters have almost identical profiles. Nothing to call them.

**Use for:** playlist seeding, "more like this" recommendations, or tagging the catalogue by mood instead of by genre.

**Don't use for:** predicting popularity. Cluster means run 29.7 to 42.7 — real, but small, and the EDA already showed no audio feature predicts a hit.

---

## 13. Limits

1. **Silhouette is 0.155.** These are regions of one cloud, not natural groups.
2. **`liveness` is noisy.** Cluster 4 caught "ROXANNE" and "The Box" — studio tracks with crowd-style ad-libs, not live recordings.
3. **PCA shows 36% of the variance.** Judge separation by the score, not the picture.
4. **Genre is a playlist label**, not a song property. Treat the crosstab as a hint, not a grade.
5. **k=6 is a choice.** k=5 scored 0.1507 against 0.1547. Close.

---

## 14. Glossary — what each thing is, and what it does

### Words

| Term | What it is | What it does |
|---|---|---|
| **clustering** | grouping rows by how similar they are | finds groups nobody labelled |
| **unsupervised** | learning with no answer key | why we never feed it `playlist_genre` |
| **k** | how many clusters you want | you must choose it, the data won't say |
| **centroid** | the centre of a cluster | it's the mean of every point inside it |
| **inertia** | total squared distance from each point to its centroid | the score K-Means minimises, see section 2 |
| **WCSS** | another name for inertia | same thing, "within-cluster sum of squares" |
| **drop** | this k's inertia subtracted from the previous k's | tells you what the extra cluster bought |
| **elbow** | the bend where the drops go flat | your first guess at k |
| **silhouette** | `(b - a) / the bigger of a, b`, averaged | scores how well-separated the clusters are |
| **a** | average distance from a point to its own cluster | small is good |
| **b** | average distance to the nearest other cluster | big is good |
| **variance** | how spread out a column is | PCA keeps the directions with the most of it |
| **explained variance ratio** | share of the total spread one component holds | tells you how much the 2D plot threw away |
| **PCA** | squashes many columns into a few | lets us draw 9 dimensions on flat paper |
| **pc1 / pc2** | the two new columns PCA builds | mixtures of all 9 features, not real features |
| **cluster profile** | the mean of every feature inside a cluster | how you work out what to name it |
| **hard assignment** | one point, one cluster | what K-Means does |
| **soft assignment** | one point, a probability per cluster | what GMM does |
| **ground truth** | a real label you can check against | ours is `playlist_genre`, used only after |

### Code

**Part 1 — Preprocessing**

| Line | What it is | What it does |
|---|---|---|
| `.dropna(subset=[...])` | pandas | removes rows missing a name, 5 of them |
| `df["duration_ms"] >= 10000` | filter | removes the 4-second track |
| `.drop_duplicates(subset="track_id")` | pandas | one row per song, 32,827 down to 28,351 |
| `.reset_index(drop=True)` | pandas | renumbers rows 0..n so the labels line up later |
| `.isna().sum().sum()` | pandas | proves 0 missing cells are left |
| `StandardScaler()` | sklearn | the scaler object |
| `.fit_transform(X)` | sklearn | learns each column's mean and std, then rewrites every value as "how many stds from average" |

**Part 2 — Choosing k**

| Line | What it is | What it does |
|---|---|---|
| `KMeans(...)` | sklearn model | the clustering algorithm |
| `n_clusters=k` | parameter | how many clusters to build |
| `random_state=42` | parameter | fixes the random start so you get the same answer every run |
| `n_init=10` | parameter | runs it 10 times from 10 random starts, keeps the lowest inertia |
| `.fit_predict(X_scaled)` | method | trains, then returns one cluster number per song |
| `.inertia_` | attribute | the inertia of the run that won |
| `silhouette_score(...)` | sklearn metric | the average silhouette across the points |
| `sample_size=5000` | parameter | scores 5,000 random points instead of all 28,351, see section 8 |

**Part 3 — Fitting**

| Line | What it is | What it does |
|---|---|---|
| `GaussianMixture(...)` | sklearn model | the second algorithm, fits overlapping ellipses |
| `n_components=k` | parameter | GMM's name for `n_clusters` |

**Part 4 — Visual inspection**

| Line | What it is | What it does |
|---|---|---|
| `PCA(n_components=2)` | sklearn | squash 9 columns down to 2 |
| `.explained_variance_ratio_` | attribute | how much spread each component kept, ours is 18.8% + 16.8% |
| `.sample(6000, random_state=1)` | pandas | 28,351 dots is a smear, 6,000 is readable |
| `hue="kmeans_cluster"` | seaborn | one colour per cluster |
| `alpha=0.6` | matplotlib | see-through dots, so overlap is visible |

**Part 5 — Interpretation**

| Line | What it is | What it does |
|---|---|---|
| `.groupby("kmeans_cluster")[features].mean()` | pandas | the cluster profile, average song per cluster |
| `scaled_frame.groupby(...).mean()` | pandas | same, but on scaled data, so the answer is in std devs |
| `sns.heatmap(..., center=0)` | seaborn | puts 0 at white, so red = above average, blue = below |
| `pd.crosstab(a, b)` | pandas | counts every combination of cluster and genre |
| `normalize="index"` | parameter | turns those counts into "% of the cluster", so big clusters don't dominate |
| `.map(cluster_names)` | pandas | swaps cluster 0..5 for readable names |
| `.groupby(...).head(3)` | pandas | top 3 rows inside each group |

**Part 6 — Comparison**

| Line | What it is | What it does |
|---|---|---|
| `.nunique()` | pandas | how many distinct clusters came out |
| `.value_counts().min() / .max()` | pandas | smallest and largest cluster, catches a model dumping everything into one |
