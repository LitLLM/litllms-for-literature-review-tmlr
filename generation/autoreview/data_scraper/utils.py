from typing import List
import pandas as pd


def write_excel_df(df_list: List, sheet_name_list: List, writer: pd.ExcelWriter = None,
                   close_writer: bool = False, save_file_path: str = None,
                   append_mode: bool = False):
    """
    Save a list of df in different sheets in one Excel file.
    Args:
        writer:
        df_list:
        sheet_name_list:
        close_writer:
        save_file_path:
        append_mode:

        
    https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without \
    -overwriting-data-using-pandas
    https://www.geeksforgeeks.org/how-to-write-pandas-dataframes-to-multiple-excel-sheets/


    Returns:
    """
    if save_file_path:
        if append_mode:
            writer = pd.ExcelWriter(save_file_path, mode="a", engine='xlsxwriter')
        else:
            writer = pd.ExcelWriter(save_file_path, engine='xlsxwriter')
    # Write each dataframe to a different worksheet
    assert len(df_list) == len(sheet_name_list)
    for index in range(len(df_list)):
        df_list[index].to_excel(writer, sheet_name=sheet_name_list[index])
    # Close the Pandas Excel writer and output the Excel file.
    if close_writer:
        writer.close()
    return
