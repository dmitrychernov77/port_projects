"""Solution for Similar Items task"""
from typing import Dict
from typing import List
from typing import Tuple

import itertools
import numpy as np
from scipy.spatial.distance import cosine


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        # Получаем все возможные пары ключей
        pairs = list(itertools.combinations(embeddings.keys(), 2))
        # Для каждой пары ключей находим косинусное сходство
        pair_sims = {(key1, key2): np.round(1 - cosine(embeddings[key1], embeddings[key2]), 8)
                    for key1, key2 in pairs}
        return pair_sims

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        # Создаем множество для хранения уникальных ключей
        unique_keys = set()

        # Добавляем все ключи из пар в множество
        for key1, key2 in sim.keys():
            unique_keys.add(key1)
            unique_keys.add(key2)

        # создаем словарь для выходных значений
        knn_dict = {}

        # для каждого уникального ключа ищем сопадения в парах и добавляем
        # список из пар и попарных похожестей до top-элемента
        for key in unique_keys:
            tmp_list = []
            # фильтруем пары по наличию ключа
            filtred_pair_sims = {pair: sim_value for pair, sim_value in sim.items() if key in pair}
            # cортируем пары значений по похожести
            sorted_pair_sims = dict(sorted(filtred_pair_sims.items(),
                                           key=lambda item: item[1], reverse=True))
            for key1, key2 in sorted_pair_sims.keys():
                if key == key1:
                    tmp_list.append((key2, sorted_pair_sims[(key1, key2)]))
                else:
                    tmp_list.append((key1, sorted_pair_sims[(key1, key2)]))
            # в листе оставляем только top элементов
            knn_dict[key] = tmp_list[:top]
        return knn_dict

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        # делаем словарь с суммами всех похожестей + 1 (чтобы веса были в интервале [0; 2])
        sum_dict = {key: np.sum([sim + 1 for _, sim in item]) for key, item in knn_dict.items()}
        # считаем средневзвешенную цену для похожих товаров, как:
        # цена * (похожесть+1) / сумму похожестей
        knn_price_dict = {key : np.round(
                                    np.sum([prices[key1] * (sim1 +1 ) / sum_dict[key]
                                            for key1, sim1 in item])
                                        , 2)
                            for key, item in knn_dict.items()}
        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        pairs_sim = SimilarItems.similarity(embeddings)
        knn_dict = SimilarItems.knn(pairs_sim, top)
        knn_price_dict = SimilarItems.knn_price(knn_dict, prices)
        return knn_price_dict
