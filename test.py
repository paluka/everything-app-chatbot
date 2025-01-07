from loguru import logger

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data
from langflow.schema.message import Message

import json


class MergeDataComponent(Component):
    """MergeDataComponent is responsible for combining multiple Data objects into a unified list of Data objects.

    It ensures that all keys across the input Data objects are present in each merged Data object.
    Missing keys are filled with empty strings to maintain consistency.
    """

    display_name = "Merge Data"
    description = (
        "Combines multiple Data objects into a unified list, ensuring all keys are present in each Data object."
    )
    icon = "merge"

    inputs = [
        DataInput(
            name="data_inputs",
            display_name="Data Inputs",
            is_list=True,
            info="A list of Data inputs objects to be merged.",
        ),
    ]

    outputs = [
        Output(
            display_name="Merged Data",
            name="merged_data",
            method="merge_data",
        ),
    ]

    def merge_data(self) -> Data:
        """Merges multiple Data objects into a single list of Data objects.

        Ensures that all keys from the input Data objects are present in each merged Data object.
        Missing keys are filled with empty strings.

        Returns:
            List[Data]: A list of merged Data objects with consistent keys.
        """
        logger.info("Initiating the data merging process.")

        data_inputs: list[Data] = self.data_inputs
        logger.debug(f"Received {len(data_inputs)} data input(s) for merging.")

        if not data_inputs:
            logger.warning("No data inputs provided. Returning an empty list.")
            return []

        # Collect all unique keys from all Data objects
        all_keys: set[str] = set()
        for idx, data_input in enumerate(data_inputs):
            if not isinstance(data_input, Data):
                error_message = f"Data input at index {idx} is not of type Data."
                logger.error(error_message)
                type_error_message = (
                    f"All items in data_inputs must be of type Data. Item at index {idx} is {type(data_input)}"
                )
                raise TypeError(type_error_message)
            all_keys.update(data_input.data.keys())
        logger.debug(f"Collected {len(all_keys)} unique key(s) from input data.")

        try:
            # Create new list of Data objects with missing keys filled with empty strings
            merged_data_dict = {}
            for idx, data_input in enumerate(data_inputs):

                for key in all_keys:
                    # Use the existing value if the key exists, otherwise use an empty string
                    value = data_input.data.get(key, "")
                    if key not in data_input.data:
                        log_message = f"Key '{key}' missing in data input at index {idx}. " "Assigning empty string."
                        logger.debug(log_message)
                        continue
                    merged_data_dict[key] = value

            merged_data = Data(
                data=merged_data_dict
            )
            
            comment_id = merged_data.comment_id
            comment_replies = merged_data.comment_replies
            
            # raise Exception(type(comment_replies[1]['in_reply_to_id']))
            
            # filtered_comments = [
            #     comment for comment in comment_replies
            #     if "in_reply_to_id" in comment and int(comment['in_reply_to_id']) == int(comment_id)
            # ]
            
            filtered_comments = []
            
            for comment in comment_replies:
                if "in_reply_to_id" not in comment:
                    continue
                if int(comment['in_reply_to_id']) == int(comment_id):
                    new_comment = { "body": comment['body'], "user": comment["user"]["login"] }
                    filtered_comments.append(new_comment)
                # else:
                #     raise Exception(f"{comment['in_reply_to_id']},{comment_id}, {type(comment['in_reply_to_id'])}, {type(comment_id)}  ")

            
            merged_data.comment_replies = filtered_comments
            
            
            logger.debug("Merged Data object created for input at index: " + str(idx))

        except Exception:
            logger.exception("An error occurred during the data merging process.")
            raise

        logger.info("Data merging process completed successfully.")
        return merged_data
