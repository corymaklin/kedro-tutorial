from kedro.pipeline import Pipeline, node, pipeline

from adult.nodes.adult import clean, split_data, train_model, evaluate_model, encode


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean,
                inputs=["raw_adult"],
                outputs="model_input_table",
                name="clean_adult_node",
            ),
            node(
                func=encode,
                inputs=["model_input_table", "params:model_options"],
                outputs="encoded_adult",
                name="encode_adult_node",
            ),
            node(
                func=split_data,
                inputs=["encoded_adult", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )