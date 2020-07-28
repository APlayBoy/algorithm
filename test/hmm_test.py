from ml.hmm import BaseHMM, LogHMM
import numpy as np
import unittest


class TestHMM(unittest.TestCase):

    def test_base_hmm(self):
        Pi = [0.2, 0.4, 0.4]
        A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
        B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
        hmm = BaseHMM(3, 2, Pi=Pi, A=A, B=B)
        obs = [0, 1, 0]
        print("base ksi: ", hmm._xi(obs, 1))
        print("base predict: ", hmm.predict(obs))
        self.assertTrue(np.round(hmm.forward_prob(obs), 5) == 0.13022)
        self.assertTrue(np.round(hmm.backward_prob(obs), 5) == 0.13022)
        self.assertTrue(np.round(hmm.fb_prob(obs), 5) == 0.13022)
        self.assertTrue(hmm.predict(obs) == [2, 2, 2])
        A = np.array([[0.8, 0.2], [0.4, 0.6]])
        B = np.array([[0.3, 0.2, 0.5], [.0, 0.8, 0.2]])
        Pi = np.array([0.2, 0.8])
        hmm = BaseHMM(2, 3, Pi, A, B)
        obs_data = hmm.gen_data(1000) #生成1000个样本
        hmm.fit(obs_data, maxstep=50) #使用生成的样本训练模型
        print(hmm.A)
        print(hmm.B)
        print(hmm.Pi)
        #允许误差为 0.25
        self.assertTrue((np.abs(A - hmm.A) < 0.25).all())
        self.assertTrue((np.abs(B - hmm.B) < 0.25).all())
        self.assertTrue((np.abs(Pi - hmm.Pi) < 0.25).all())

    def test_log_hmm(self):
        Pi = [0.2, 0.4, 0.4]
        A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
        B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
        hmm = LogHMM(3, 2, Pi=np.log(Pi), A=np.log(A), B=np.log(B))
        obs = [0, 1, 0]
        print("log ksi: ", hmm._xi(obs, 1))
        print("log predict: ", hmm.predict(obs))
        self.assertTrue(np.round(hmm.forward_prob(obs), 5) == 0.13022)
        self.assertTrue(np.round(hmm.backward_prob(obs), 5) == 0.13022)
        self.assertTrue(np.round(hmm.fb_prob(obs), 5) == 0.13022)
        self.assertTrue(hmm.predict(obs) == [2, 2, 2])

        A = np.array([[0.8, 0.2], [0.4, 0.6]])
        B = np.array([[0.3, 0.2, 0.5], [1e-5, 0.8-1e-5, 0.2]])
        Pi = np.array([0.2, 0.8])
        hmm = LogHMM(2, 3, np.log(Pi), np.log(A), np.log(B))
        obs_data = hmm.gen_data(1000) #生成1000个样本
        hmm.fit(obs_data, maxstep=50) #使用生成的样本训练模型
        print(np.exp(hmm.A))
        print(np.exp(hmm.B))
        print(np.exp(hmm.Pi))
        # #允许误差为 0.25
        self.assertTrue((np.abs(A - np.exp(hmm.A)) < 0.25).all())
        self.assertTrue((np.abs(B - np.exp(hmm.B)) < 0.25).all())
        self.assertTrue((np.abs(Pi - np.exp(hmm.Pi)) < 0.25).all())


if __name__ == '__main__':
    unittest.main()
