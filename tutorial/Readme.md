## Theory:
* [论文浅尝 | Complex Embeddings for Simple Link Prediction](https://blog.csdn.net/tgqdt3ggamdkhaslzv/article/details/79081541)
* [图谱论文笔记2 - ComplEx](https://longaspire.github.io/blog/%E5%9B%BE%E8%B0%B1%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B02/)
* [怎么理解虚数和复数？](https://zhuanlan.zhihu.com/p/350085395)

## Video:
* [國產視頻](https://search.bilibili.com/all?keyword=%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%EF%BC%88Knowledge%20Graph)
* [Official Site](https://docs.ampligraph.org/en/1.3.2/tutorials.html)

## Reference:
* [AmpliGraph初步实践](https://juejin.cn/post/7033386911968428040)
* [ECAI 2020 Tutorials](https://www.youtube.com/watch?v=gX_KHaU8ChI)

## Note:
1. Hermitian dot product(複數內積 - 後項都要取bar)
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
