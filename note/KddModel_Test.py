import unittest
import KddModel as kdd


class Kddtest(unittest.TestCase):
    def setUp(self):
        print("test start...")
        self.testdata = kdd.nsl_csv_read('data/nsl-kdd/KDDTrain+.txt')

    def test_kddset(self):
        for i in range(5):
            data = self.testdata[i]
            print(data.shape)
            print(data)
            data, label = kdd.nslconvert(data)
            print(data.shape)
            print(label)

    def tearDown(self):
        print("test end...")


if __name__ == '__main__':
    unittest.main()
