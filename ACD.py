import sys
sys.path.append(r'C:\\Users\\Owner\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\')
import matlab.engine


class ACD:

    @staticmethod
    def increase_monotonicity(signal):
        if len(signal) <= 1:
            return signal
        eng = matlab.engine.start_matlab()
        inp = matlab.double(list(signal))
        ret = eng.trendacd(inp)[0]
        return ret