import json
import os
import sys

def verify():
    print("Verification Step 1: Simulating verification environment (renaming training.csv)...")
    has_training = os.path.exists('./dataset/training.csv')
    if has_training:
        os.rename('./dataset/training.csv', './dataset/training.csv.bak')
        print("Temporarily renamed training.csv to training.csv.bak.")
    else:
        print("training.csv is already not present (simulated environment).")

    try:
        print("\nVerification Step 2: Reading pipeline.ipynb cells...")
        with open('pipeline.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)

        code_lines = []
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell_code = "".join(cell['source'])
                # Filter out IPython magic commands
                filtered_lines = []
                for line in cell_code.splitlines():
                    if line.strip().startswith('%') or line.strip().startswith('!'):
                        continue
                    filtered_lines.append(line)
                code_lines.append("\n# --- NEW CELL ---\n" + "\n".join(filtered_lines))

        full_code = "\n".join(code_lines)

        print("\nVerification Step 3: Executing notebook code...")
        # Execute the python code in a separate namespace
        global_namespace = {}
        exec(full_code, global_namespace)
        print("Notebook code executed successfully!")

        print("\nVerification Step 4: Checking output files...")
        if os.path.exists('predicted.csv') and os.path.exists('predicted_demand.csv'):
            pred_df = pd = global_namespace.get('pd').read_csv('predicted.csv')
            print(f"predicted.csv exists and has shape {pred_df.shape}!")
            print("First 5 rows:")
            print(pred_df.head())
            print("\nVERIFICATION PASSED! The fallback pipeline works flawlessly without training.csv.")
        else:
            print("ERROR: Output files predicted.csv or predicted_demand.csv were not created!")
            sys.exit(1)

    except Exception as e:
        print(f"\nVERIFICATION FAILED: An error occurred during notebook execution:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if has_training and os.path.exists('./dataset/training.csv.bak'):
            os.rename('./dataset/training.csv.bak', './dataset/training.csv')
            print("\nRestored training.csv from training.csv.bak.")

if __name__ == "__main__":
    verify()
