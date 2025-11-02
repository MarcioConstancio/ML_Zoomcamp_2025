import pickle
import pandas as pd
import sys

# --- Configuração ---

# O nome do arquivo do pipeline que queremos carregar
MODEL_FILENAME = 'pipeline_v1.bin'

# O registro que queremos pontuar
# Nota: "number_of_courses_vieved" foi corrigido para "number_of_courses_viewed"
# para corresponder às features esperadas pelo pipeline.
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# --- Funções ---

def load_pipeline(filename):
    """
    Carrega um pipeline serializado (pickle) a partir de um arquivo binário.
    """
    try:
        # Abre o arquivo no modo 'read binary' (rb)
        with open(filename, 'rb') as f_in:
            # Deserializa o objeto
            pipeline = pickle.load(f_in)
        print(f"Sucesso: Pipeline carregado de '{filename}'")
        return pipeline
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo não encontrado em '{filename}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Erro ao carregar o pipeline: {e}", file=sys.stderr)
        return None

def score_record(record, pipeline):
    """
    Pontua um único registro usando o pipeline carregado.
    """
    # Converte o dicionário do registro em um DataFrame do Pandas,
    # pois os pipelines do Scikit-Learn esperam uma entrada 2D.
    try:
        X = pd.DataFrame([record])
        
        # Usa .predict_proba() para obter a "pontuação" (probabilidade)
        # O resultado é [prob_classe_0, prob_classe_1]
        probabilities = pipeline.predict_proba(X)[0]
        
        # A "pontuação" é geralmente a probabilidade da classe positiva (classe 1)
        score = probabilities[1]
        
        # Também podemos obter a previsão de classe direta (0 ou 1)
        prediction = pipeline.predict(X)[0]
        
        return score, prediction, probabilities
        
    except KeyError as e:
        print(f"Erro: A feature {e} não foi encontrada.", file=sys.stderr)
        print("Verifique se as chaves do registro correspondem às features do pipeline.", file=sys.stderr)
        return None, None, None
    except Exception as e:
        print(f"Erro durante a pontuação: {e}", file=sys.stderr)
        return None, None, None

# --- Execução Principal ---

if __name__ == "__main__":
    # 1. Carregar o pipeline
    pipeline = load_pipeline(MODEL_FILENAME)
    
    if pipeline:
        # 2. Pontuar o registro
        print(f"\nPontuando o registro:\n{record}")
        score, prediction, probabilities = score_record(record, pipeline)
        
        if score is not None:
            # 3. Exibir os resultados
            print("\n--- Resultados da Pontuação ---")
            print(f"Probabilidades (Classe 0, Classe 1): {probabilities}")
            print(f"Previsão da Classe: {prediction}")
            print(f"Pontuação (Probabilidade da Classe 1): {score:.4f}")
