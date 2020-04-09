import unittest
import KddModel


class Kddtest(unittest.TestCase):
    def setUp(self):
        print("test start...")
        self.testdata = KddModel.nsl_csv_read(
            '../../data/kdd/kddcup10.corrected')
        self.kddset = KddModel.Kddset(
            'kddcup10.corrected', root='../../data/kdd/')

    def test_kddset(self):
        for i in range(5):
            data, label = self.kddset[i]
            print(data.shape)
            print(label)

    def tearDown(self):
        print("test end...")


if __name__ == '__main__':
    unittest.main()
