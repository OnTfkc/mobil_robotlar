# KNN Destekli RRT Yol Planlamada Daha Yumuşak Geçişli Yol Üretimi için Eğri Tabanlı İyileştirme Yaklaşımları

Bu proje, **KNN-RRT (k-Nearest Neighbor Rapidly-exploring Random Tree)** algoritması çıktısı olduğu varsayılan düzensiz (raw) yol noktalarını,
farklı yumuşatma (smoothing) teknikleri kullanarak daha düzgün ve takip edilebilir bir yol haline getirmeyi amaçlar.

Projede:
- **Bezier eğrisi**
- **Engel farkındalıklı B-Spline**

yaklaşımları karşılaştırmalı olarak uygulanmıştır.

---

## İçerik Özeti

- Raw (KNN-RRT) yol noktalarının tanımlanması  
- Bezier eğrisi ile global yol yumuşatma  
- Engellere yakın noktalara itme (repulsion) uygulanması  
- B-Spline ile yerel ve pürüzsüz yol üretimi  
- Tüm yolların ve engellerin görselleştirilmesi  

---

## Kullanılan Kütüphaneler

- `numpy`
- `matplotlib`
- `scipy`
- `math`

Kurulum için:
```bash
pip install numpy matplotlib scipy
