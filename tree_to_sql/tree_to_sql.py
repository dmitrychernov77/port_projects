import json
from sklearn.tree import DecisionTreeClassifier


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    """рекурсивно разбираем дерево в json"""
    tmp_tree = tree.tree_
    def tree_to_dict(node_index):
        result = {}
        is_leaf = tmp_tree.children_left[node_index] == -1 and tmp_tree.children_right[node_index] == -1

        if is_leaf:
            # It's leaf
            class_label = int(tmp_tree.value[node_index].argmax())
            result["class"] = class_label
        else:
            # It's not leaf
            result['feature_index'] = int(tmp_tree.feature[node_index])
            result['threshold'] = round(float(tmp_tree.threshold[node_index]), 4)
            left_child_index = tmp_tree.children_left[node_index]
            right_child_index = tmp_tree.children_right[node_index]
            result['left'] = tree_to_dict(left_child_index)
            result['right'] = tree_to_dict(right_child_index)
        return result
    tree_as_json = json.dumps(tree_to_dict(0))
    return tree_as_json


def generate_sql_query(tree_as_json: str, features: list) -> str:
    """рекурсивно собираем json в sql-запрос"""
    data = json.loads(tree_as_json)
    feature_name = features[data['feature_index']]
    treshold = data['threshold']

    def dict_tree_to_sql(data: dict, feature_name: str, treshold: float) -> str:
        if 'class' not in data.keys():
            feature_name = features[data['feature_index']]
            treshold = data['threshold']
            left_node = data['left']
            right_node = data['right']

            left_sql = dict_tree_to_sql(left_node, feature_name, treshold)
            right_sql = dict_tree_to_sql(right_node, feature_name, treshold)
            return f"CASE WHEN {feature_name} > {treshold} THEN {right_sql} ELSE {left_sql} END"
        else:
            return f"{data['class']}"

    query = "SELECT " + dict_tree_to_sql(data, feature_name, treshold) + " AS CLASS_LABEL"
    return query
