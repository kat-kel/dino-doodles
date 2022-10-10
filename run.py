import click
from prep_data import Data
from topic_modeling import tf_idf

@click.command()
@click.option("--max_df", default=0.67, help="(float) threshold at which to ignore words that occur in that percentage of documents or more")
@click.option("--min_df", default=3, help="(integer) threshold at which to ignore words that occur that number of times or fewer in the corpus")
@click.option("--clusters", default=6, help="(integer) number of clusters to create")
@click.option("--members", default=8, help="(integer) number of terms to be included in one cluster")
def main(max_df, min_df, clusters, members):
    MAX_DF = max_df
    MIN_DF = min_df
    NO_OF_CLUSTERS = clusters
    NO_OF_MEMBERS = members
    cleaned_data = [d["Cleaned Proposition"] for d in Data().clean()]
    result = tf_idf(cleaned_data, MAX_DF, MIN_DF, NO_OF_CLUSTERS, NO_OF_MEMBERS)
    for i, cluster in enumerate(result):
        print("-----")
        print(f"Cluster {i+1}:")
        print(cluster)
        print("")

if __name__ == "__main__":
    main()