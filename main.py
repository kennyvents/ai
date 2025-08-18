from utils.data_utils import read_data, normalize_data, get_result
from utils.match_finder import find_top_matches
from utils.excel_log import get_top_ent

data = read_data('modified_prices', 'заявка')
data = normalize_data(data[0], data[1])

if __name__ == '__main__':
    result_data = find_top_matches(data[0], data[1], data[2])
    get_result(result_data)
    get_top_ent('files/results/emb_results', 'files/results/finally_results')
