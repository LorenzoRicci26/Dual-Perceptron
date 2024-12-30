Tutti i test fatti su un addestramento di 1000 epoche 

1)Breast Cancer Wisconsin (Diagnostic)

n_samples= 569 | n_features = 30

1.1) Linear: Time taken to train the data is 211.1031446000561s
             Time taken to predict the target is 0.048991099931299686s
             Accuracy sul set di test: 0.9649122807017544

2.1)Polinomial: (degree=2): Time taken to train the data is 317.7160481000319s
                            Time taken to predict the target is 0.0651627000188455s
                            Accuracy sul set di test: 0.9473684210526315
                (degree=3): Time taken to train the data is 7.738842600025237s
                            Time taken to predict the target is 0.06485279998742044s
                            Accuracy sul set di test: 0.956140350877193
                (degree=4): Time taken to train the data is 13.973362900083885s
                            Time taken to predict the target is 0.06720710010267794s
                            Accuracy sul set di test: 0.9385964912280702
                (degree=5): Time taken to train the data is 43.87366689997725s
                            Time taken to predict the target is 0.0663301000604406s
                            Accuracy sul set di test: 0.9210526315789473

2.3)Radial Basis Function (RBF):(gamma=1):  Time taken to train the data is 2.5056451000273228s
                                            Time taken to predict the target is 0.15058159991167486s
                                            Accuracy sul set di test: 0.9385964912280702
                                (gamma=2):  Time taken to train the data is 2.4862157000461593s
                                            Time taken to predict the target is 0.14933080004993826s
                                            Accuracy sul set di test: 0.9298245614035088
                                (gamma=3):  Time taken to train the data is 2.4455308000324294s
                                            Time taken to predict the target is 0.15411410003434867s
                                            Accuracy sul set di test: 0.9385964912280702
                                (gamma=0.1):Time taken to train the data is 4.356686300016008s
                                            Time taken to predict the target is 0.14728079992346466s
                                            Accuracy sul set di test: 0.9473684210526315
                                (gamma=0.5):Time taken to train the data is 2.5589759000577033s
                                            Time taken to predict the target is 0.15575069992337376s
                                            Accuracy sul set di test: 0.9649122807017544
                                (gamma=0.3):Time taken to train the data is 2.4690503999590874s
                                            Time taken to predict the target is 0.14887229993473738s
                                            Accuracy sul set di test: 0.9649122807017544

2)Adult 

n_samples= 1000 | n_features = 7 identificate come le più rilevanti all'interno del dataset

Linear-Kernel:Time taken to train the data is 537.7129719000077s
              Time taken to predict the target is 0.21054480003658682s
              Accuracy sul set di test: 0.74
(3000 samples)Time taken to train the data is 6540.7813246999995s
              Time taken to predict the target is 1.4751514999989013s
              Accuracy sul set di test: 0.7366666666666667

Polynomial_Kernel:(degree=2):Time taken to train the data is 604.7094605000457s
                             Time taken to predict the target is 0.2401421999093145s
                             Accuracy sul set di test: 0.7366666666666667, (874.3212939000223s)--->0.76
                  (degree=3):Time taken to train the data is 623.953101699939s
                             Time taken to predict the target is 0.24338980007451028s
                             Accuracy sul set di test: 0.7533333333333333 (887.771516699926s)--->0.69
                  (degree=4):Time taken to train the data is 753.4036258999258s
                             Time taken to predict the target is 0.18600609991699457s
                             Accuracy sul set di test: 0.58
                  (degree=5):Time taken to train the data is 759.0900977999845s
                             Time taken to predict the data is 0.17594310001004487s
                             Accuracy on the test set is 0.55

RBF_kernel =(gamma=1):Time taken to train the data is 2050.8801598999416s
                      Time taken to predict the target is 0.4613468999741599s
                      Accuracy sul set di test: 0.76 
            (gamma=2):Time taken to train the data is 1615.423785699997s
                      Time taken to predict the target is 0.6563004999188706s
                      Accuracy sul set di test: 0.75 
            (gamma=3):Time taken to train the data is 1930.963743899949s
                      Time taken to predict the target is 0.5015923999017105s
                      Accuracy sul set di test: 0.765
            (gamma=0.1):Time taken to train the data is 3601.8291987000266s
                        Time taken to predict the target is 0.4583304999396205s
                        Accuracy sul set di test: 0.775

3)Rice (Cammeo o Osrick) 

n_samples= 3810 | n_features = 7

3.1) Linear_kernel =  Time taken to train the data is 8178.576360200066s
                      Time taken to predict the target is 2.7262883000075817s
                      Accuracy sul set di test: 0.8705161854768154

3.2) Polynomial_kernel:(degree=2):Time taken to train the data is 24.11233719997108s
                                  Time taken to predict the target is 2.445953299989924s
                                  Accuracy: 1.0
                       (degree=3):Time taken to train the data is 24.91766000003554s
                                  Time taken to predict the target is 2.617722800001502s
                                  Accuracy sul set di test: 0.9251968503937008
                       (degree=4):Time taken to train the data is 25.220657699974254s
                                  Time taken to predict the target is 2.6191924000158906s
                                  Accuracy sul set di test: 1.0
                       (degree=5):Time taken to train the data is 24.42691040004138s
                                  Time taken to predict the target is 2.5602625000756234s
                                  Accuracy sul set di test: 0.8031496062992126

3.3) RBF_kernel:(gamma=1):Time taken to train the data is 53.72690450004302s
                          Time taken to predict the target is 5.9136080999160185s
                          Accuracy sul set di test: 1.0
                (gamma=2):Time taken to train the data is 58.0286660999991s
                          Time taken to predict the target is 6.946406499948353s
                          Accuracy sul set di test: 1.0
                (gamma=0.1):Time taken to train the data is 66.91749470005743s
                            Time taken to predict the target is 6.665008899988607s
                            Accuracy sul set di test: 1.0
                (gamma=0.5):Time taken to train the data is 59.78880349989049s
                            Time taken to predict the target is 6.908514400012791s
                            Accuracy sul set di test: 1.0

4) KR-vs-KP:

n_samples= 3195 | n_features = 36

4.1) Linear:Time taken to train the data is 7245.070811099955s
            Time taken to predict the target is 1.4252164999488741s
            Accuracy sul set di test: 0.9687010954616588

4.2) Polynomial:(degree=2):Time taken to train the data is 301.6706443000585s
                           Time taken to predict the target is 2.1262137000449s
                           Accuracy sul set di test: 0.9671361502347418
                (degree=3):Time taken to train the data is 100.28727450000588s
                           Time taken to predict the target is 2.0810958000365645s
                           Accuracy sul set di test: 0.9702660406885759
                (degree=4):Time taken to train the data is 77.35616760002449s
                           Time taken to predict the target is 4.865526100038551s
                           Accuracy sul set di test: 0.971830985915493
                (degree=5):Time taken to train the data is 140.83056190004572s
                           Time taken to predict the target is 1.9455210000742227s
                           Accuracy sul set di test: 0.971830985915493

4.3) RBF:(gamma=1): Time taken to train the data is 91.37280770007055s
                    Time taken to predict the target is 4.3075353999156505s
                    Accuracy sul set di test: 0.9280125195618153 
         (gamma=2): Time taken to train the data is 73.27573130000383s
                    Time taken to predict the target is 4.365924600046128s
                    Accuracy sul set di test: 0.9280125195618153
        (gamma=0.1):Time taken to train the data is 117.32166829996277s
                    Time taken to predict the target is 4.2910485999891534s
                    Accuracy sul set di test: 0.9624413145539906
        (gamma=0.2):Time taken to train the data is 169.90422180003952s
                    Time taken to predict the target is 4.3553756000474095s
                    Accuracy sul set di test: 0.9687010954616588
        (gamma=0.15):Time taken to train the data is 86.72640460007824s
                     Time taken to predict the target is 4.159189600031823s
                     Accuracy sul set di test: 0.9687010954616588