from Bio import SeqIO
import os
import csv
import matplotlib.pyplot as plt

# Parâmetros
pasta = "./"                # Pasta onde estão os .ab1
limite_qualidade = 20       # Corte de qualidade (Phred)
min_tamanho = 800           # Mínimo de bp para considerar boa
pasta_boas = "sequencias_boas"  # Pasta para salvar FASTAs bons
pasta_graficos = "graficos_qualidade"

# Função para cortar sequência com base na qualidade
def trim_por_qualidade(seq, qual, limite=20):
    inicio = 0
    fim = len(seq)

    while inicio < fim and qual[inicio] < limite:
        inicio += 1
    while fim > inicio and qual[fim - 1] < limite:
        fim -= 1
    return seq[inicio:fim]

# Criar pastas de saída
os.makedirs(pasta_boas, exist_ok=True)
os.makedirs(pasta_graficos, exist_ok=True)

# Arquivos de saída
fasta_bruto = "sequencias_brutas.fasta"
fasta_limpo = "sequencias_limpas.fasta"
relatorio_csv = "relatorio_qualidade.csv"

with open(fasta_bruto, "w") as fb, open(fasta_limpo, "w") as fl, open(relatorio_csv, "w", newline="") as rcsv:
    writer = csv.writer(rcsv)
    writer.writerow(["Arquivo", "Tamanho_bruto", "Tamanho_limpo", "Qualidade_media", "Status"])

    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith(".ab1"):
            caminho = os.path.join(pasta, arquivo)
            try:
                registro = SeqIO.read(caminho, "abi")

                seq_bruta = registro.seq
                tam_bruto = len(seq_bruta)

                qual = registro.letter_annotations["phred_quality"]
                qual_media = round(sum(qual) / len(qual), 2) if qual else 0

                seq_limpa = trim_por_qualidade(seq_bruta, qual, limite=limite_qualidade)
                tam_limpo = len(seq_limpa)

                # Salvar no FASTA (bruto e limpo)
                fb.write(f">{arquivo}_bruto\n{seq_bruta}\n")
                fl.write(f">{arquivo}_limpo\n{seq_limpa}\n")

                # Determinar status
                if tam_limpo >= min_tamanho and qual_media >= limite_qualidade:
                    status = "BOA"
                    with open(os.path.join(pasta_boas, f"{arquivo}_limpo.fasta"), "w") as f_out:
                        f_out.write(f">{arquivo}_limpo\n{seq_limpa}\n")
                else:
                    status = "RUIM"

                # Salvar no relatório
                writer.writerow([arquivo, tam_bruto, tam_limpo, qual_media, status])

                # Criar gráfico de qualidade
                plt.figure(figsize=(10, 4))
                plt.plot(qual, label="Qualidade por base")
                plt.axhline(y=limite_qualidade, color="red", linestyle="--", label=f"Corte Q{limite_qualidade}")
                plt.xlabel("Posição na leitura (bp)")
                plt.ylabel("Score de Qualidade (Phred)")
                plt.title(f"Perfil de qualidade: {arquivo}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(pasta_graficos, f"{arquivo}.png"))
                plt.close()

                print(f"✅ {arquivo} | Bruto: {tam_bruto} bp | Limpo: {tam_limpo} bp | Q média: {qual_media} | {status}")

            except Exception as e:
                print(f"❌ Erro ao processar {arquivo}: {e}")

print("\n🎯 Processamento concluído!")
print(f"- Sequências brutas: {fasta_bruto}")
print(f"- Sequências limpas: {fasta_limpo}")
print(f"- Relatório: {relatorio_csv}")
print(f"- Sequências boas: {pasta_boas}")
print(f"- Gráficos de qualidade: {pasta_graficos}")
