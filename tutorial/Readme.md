## Note:
### 1. Hermitian dot product(複數內積 - 後項都要取bar)
```
x = (1+i, 2+3i) , y = (2+i, 3+4i)
                 _____         ______
  <x , y> = (1+i)(2+i) + (2+3i)(3+4i)
          = (1+i)(2-i) + (2+3i)(3-4i)
          = (2-i + 2i+1) + (6-8i + 9i+12)
          = (3+i) + (18+i)
          = 21+2i

Field = R (real - 實數空間), 顯然取不取bar皆是一致的；
Field = C (complex - 複數空間), 若不取bar則無法得到一內積空間,

 ∵〈x,x〉＞ 0 and〈ix,ix〉= -1〈x,x〉＞ 0, x≠0  => contradiction.
 ```
 
### 2. ComplEx的Embedding Initialize
1. latent_features/initializers.py 
- 初始化ent_emb(entity embedding)與rel_emb(relation embedding)的原始碼
```python
def _initialize_parameters(self):
  ...
  else:
  # initialize entity embeddings to zero (these are reinitialized every batch by batch embeddings)
  # entity 初始化: tf.zeros_initializer()
  self.ent_emb = tf.get_variable('ent_emb_{}'.format(timestamp),
                                 shape=[self.batch_size * 2, self.internal_k],
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32)
                                 
  # relation 初始化: tf.random_normal_initializer()
  self.rel_emb = tf.get_variable('rel_emb_{}'.format(timestamp),
                                 shape=[len(self.rel_to_idx), self.internal_k],
                                 initializer=self.initializer.get_relation_initializer(
                                 len(self.rel_to_idx), self.internal_k),
                                 dtype=tf.float32)
                                 
    ->  def get_relation_initializer(self, in_shape=None, out_shape=None, init_type='tf'):
            """ Initializer for relation embeddings
            Returns
            -------
            initialized_values: tf.Op or n-d array
                Weights initializer
            """
            assert init_type in ['tf', 'np'], 'Invalid initializer type!'
            if init_type == 'tf':
                return self._get_tf_initializer(in_shape, out_shape, 'r')
            else:
                return self._get_np_initializer(in_shape, out_shape, 'r')

      ->  def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
            """Create a tensorflow node for initializer
            Returns
            -------
            initializer_instance: An Initializer instance.
            """
            return tf.random_normal_initializer(mean=self._initializer_params['mean'],
                                                stddev=self._initializer_params['std'],
                                                dtype=tf.float32)

```

2. latent_features/models/EmbeddingModel.py 
- 將ent_emb與rel_emb透過tf.nn.embedding_lookup方法轉成要計算loss的e_s, e_p, e_o
- e_s, e_p, e_o會再透過tf.split切成scoring function的實數虛數兩個向量

```python
def _lookup_embeddings(self, x, get_weight=False):
"""Get the embeddings for subjects, predicates, and objects of a list of statements used to train the model.
        Parameters
        ----------
        x : tensor, shape [n, k]
            A tensor of k-dimensional embeddings
        """
        e_s = self._entity_lookup(x[:, 0])
        e_p = tf.nn.embedding_lookup(self.rel_emb, x[:, 1])
        e_o = self._entity_lookup(x[:, 2])
        
        if get_weight:
            wt = self.weight_triple[
                self.batch_number * self.batch_size:(self.batch_number + 1) * self.batch_size]
        
            return e_s, e_p, e_o, wt
        return e_s, e_p, e_o
```

3.參數更新:
- 取每個batch的avg loss- > f(scoring), 透過sgd更新


### 3. Scoring function of ComplEx:
直覺上, 當s&o可逆, 我希望他的score是最大的 / 
當不可逆的時候, score可以思考為: s指向的影響力-o反指向的影響力
```python
source code from ./model/ComplEx.py
----------
# Assume each embedding is made of an img and real component.
# (These components are actually real numbers, see [trouillon2016complex].
e_s_real, e_s_img = tf.split(e_s, 2, axis=1)
e_p_real, e_p_img = tf.split(e_p, 2, axis=1)
e_o_real, e_o_img = tf.split(e_o, 2, axis=1)

# See Eq. 9 [trouillon2016complex):
return tf.reduce_sum(e_p_real * e_s_real * e_o_real, axis=1) + \  # 可逆的話我希望s,o的實部內積越大越好
    tf.reduce_sum(e_p_real * e_s_img * e_o_img, axis=1) + \       # 可逆的話我希望s,o的虛部內積越大越好
    tf.reduce_sum(e_p_img * e_s_real * e_o_img, axis=1) - \       # 不可逆的話我希望s實部指向o的虛部內積越大越好
    tf.reduce_sum(e_p_img * e_s_img * e_o_real, axis=1)           # 不可逆的話我希望s虛部指向o的實部內積越小越好
```

### 4. ComplEx的Score funcion 
```
Case1: 兩實體是雙向, 則score要大
Case2: E1指向E2, Score要大 / E2反指向E1不成立, Score小
```
* Insight from AmpliGraph
1. The  <Re(Wr),Re(Es),Re(Eo)>  + <Re(Wr),Im(Es),Im(Eo)> can handle symmertic relations and the last two terms can handle anti-symmetry. Ideally, for symmertic relations Im(Wr) should be 0 and for anti-symmetric relations Im(Wr) !=0 

2. Example of Symmetric relation is colleague_with. i.e. if <Sumit, colleague_with, Luca> then <Luca, colleague_with, Sumit> is also true. Both these triples should get same score. The first 2 terms of the above equation ensures this if im(Wr)==0.. When im(Wr) is 0, the complex score is Re( colleague_with) * Re(Sumit) * Re(Luca) + Re(colleague_with) * Im(Sumit) * Im(Luca) for the first  triple as well as second triple. i.e. the score is same.

3. When im(Wr) !=0, it can handle anti-symmetric relations. Eg of anti-symmetric relation is lives_in: <Sumit lives_in, Dublin> doesnt imply <Dublin, lives_in, Sumit>. If the first triple get's a high score, the second one should get a low score. This is ensured by the above equation when Im(Wr) !=0. For the above 2 triples, the first 2 terms of the complex scoring function is same. The last two term for <Sumit lives_in, Dublin>  is Im(lives_in) * ( Re(Sumit)*Im(Dublin) - Im(Sumit)Re(Dublin)  ) whereas for <Dublin, lives_in, Sumit> it is  Im(lives_in) * (  Im(Sumit)*Re(Dublin)- Re(Sumit)*Im(Dublin)  ). Hence if first triple gets a high score, the second one will get a low score. 

### 5. ComplEx的loss - arg minΘ∑r(s,o)∈Ωlog(1+exp(−Yrsoϕ(s,r,o;Θ)))+λ||Θ||22
* 當score很高,exp(-score)很小 -> 原式趨近於 arg min log(1) -> 趨近0, 代表沒有loss, 完美預測 
* 當score很小甚至是負數,exp(-score)很大 -> 原式趨近於 arg min log(1+一個很大的數字) -> loss巨大, gradient走起 

## Topics:
* [知乎 | Complex Embeddings for Simple Link Prediction](https://zhuanlan.zhihu.com/p/107914673)
* [论文浅尝 | Complex Embeddings for Simple Link Prediction](https://blog.csdn.net/tgqdt3ggamdkhaslzv/article/details/79081541)
* [图谱论文笔记2 - ComplEx](https://longaspire.github.io/blog/%E5%9B%BE%E8%B0%B1%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B02/)
* [怎么理解虚数和复数？](https://zhuanlan.zhihu.com/p/350085395)
* [负对数似然(negative log-likelihood)](https://blog.csdn.net/silver1225/article/details/88914652)

## Videos:
* [國產視頻](https://search.bilibili.com/all?keyword=%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%EF%BC%88Knowledge%20Graph)
* [Official Site](https://docs.ampligraph.org/en/1.3.2/tutorials.html)

## Reference:
* [AmpliGraph初步实践](https://juejin.cn/post/7033386911968428040)
* [ECAI 2020 Tutorials](https://www.youtube.com/watch?v=gX_KHaU8ChI)
