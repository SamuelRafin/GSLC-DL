| Kode | Strategi | Trainable Params | Learning Rate | Epoch Konvergen (best) | Epoch Dijalankan | Waktu Latih (s) | Acc Train (clean) | Acc Val | Gap Train-Val | Acc Test | Macro F1 Test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Baseline from scratch | 11,181,642 | all=0.001 | 11 | 15 | 230.1 | 0.6149 | 0.52 | 0.0949 | 0.52 | 0.5045 |
| B2 | Feature extraction (head saja) | 5,130 | head=0.001 | 19 | 23 | 320.5 | 0.9843 | 0.97 | 0.0143 | 0.97 | 0.9699 |
| B3 | Progressive fine-tuning (layer4+fc, differential LR) | 8,398,858 | last_block=0.0001, head=0.001 | 9 | 13 | 184.4 | 0.9993 | 0.98 | 0.0193 | 0.9533 | 0.9532 |
| B4-1 | B3 TANPA augmentasi | 8,398,858 | last_block=0.0001, head=0.001 | 19 | 23 | 229.1 | 1.0 | 0.99 | 0.01 | 0.96 | 0.9599 |
| B4-2 | B3 DENGAN augmentasi (= B3) | 8,398,858 | last_block=0.0001, head=0.001 | 9 | 13 | 184.4 | 0.9993 | 0.98 | 0.0193 | 0.9533 | 0.9532 |
| B5-efficientnet_b0 | efficientnet_b0 (ft) | 1,142,202 | last_block=0.0001, head=0.001 | 11 | 15 | 213.8 | 0.9971 | 0.9967 | 0.0005 | 0.9767 | 0.9767 |