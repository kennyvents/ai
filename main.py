from utils.data_utils import read_data, normalize_data, get_result
from utils.match_finder import find_top_matches

data = read_data('modified_prices', 'Закупка')
data = normalize_data(data[0], data[1])

if __name__ == '__main__':
    result_data = find_top_matches(data[0], data[1])
    get_result(result_data)
