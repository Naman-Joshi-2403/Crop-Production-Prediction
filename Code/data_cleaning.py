import pandas as pd
from dotenv import load_dotenv
import os

login_user = os.path.expanduser("~")
load_dotenv("dev.env")

input_dataset_path = os.getenv("INPUT_DATASET_PATH")
clean_csv_output_path = os.getenv("OUTPUT_DATASET_PATH")

class DataCleaning:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def load_dataset(self):
        df = pd.read_excel(self.input_path)
        return df

    def basic_cleaning(self, df):
        required_columns = ["Area",
                            "Item",
                            "Year",
                            "Element",
                            "Unit",
                            "Value",
                            "Flag Description"
                        ]
        
        df = df[required_columns]
        df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_"))

        ## removing values if falg or value columns are empty 
        df = df.dropna(subset=["value"])
        df = df.dropna(subset=["flag_description"])
        return df

    def adding_confidence_score(self, df):
        confidence_score_mapping = {
            "Official figure": 1.0,
            "Figure from international organizations": 1.0,
            "Estimated value": 0.7,
            "Imputed value": 0.6
        }
        df['confidence_score'] = df['flag_description'].map(confidence_score_mapping)
        df = df.dropna(subset=["confidence_score"])

        return df

    def filtering_required_element(self, df):
        required_element = [ 
            "Area harvested",
            "Yield",
            "Production"
        ]

        df = df[df['element'].isin(required_element)]
        
        return df

    def validating_units(self, df):
        validating_units = {
            "Area harvested": "ha",
            "Yield": "kg/ha",
            "Production": "t"
        }

        df = df [
            df.apply(
                lambda row : row['unit'] == validating_units.get(row['element']),
                axis = 1
            )
        ]

        return df

    def pivot_elements(self, df):
        df_pivot = df.pivot_table(
            index = ["area", "item", "year"],
            columns = "element",
            values = ["value", "confidence_score"],
            aggfunc = "mean"
        )

        df_pivot.columns = [
            f"{col[1].lower().replace(' ', '_')}"
            if col[0] == "value"
            else "confidence_score"
            for col in df_pivot.columns
        ]
        
        df_pivot = df_pivot.reset_index()

        df_pivot = df_pivot.rename(columns={
            "area_harvested": "area_harvested_ha",
            "yield": "yield_kg_per_ha",
            "production": "production_tons"
        })

        return df_pivot

    def handel_missing_value(self, df):
        df = df.dropna(subset=['production_tons'])
        feature_cols = [
            "area_harvested_ha",
            "yield_kg_per_ha",
            "confidence_score"
        ]

        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())

        return df

    def save_dataset(self, df):
        df.to_csv(self.output_path, index=False)

    def main(self):
            df = self.load_dataset()
            df = self.basic_cleaning(df)
            df = self.adding_confidence_score(df)
            df = self.filtering_required_element(df)
            df = self.validating_units(df)
            df = self.pivot_elements(df)
            df = self.handel_missing_value(df)
            self.save_dataset(df)


if __name__ == "__main__":
    input_path = os.path.join(login_user, input_dataset_path)
    output_path = os.path.join(login_user, clean_csv_output_path)
    cleaner = DataCleaning(input_path, output_path)
    cleaner.main()
