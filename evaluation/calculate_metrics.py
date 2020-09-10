from evaluate_generation import *
import os
import json
from pathlib import Path

from fairseq.models.transformer import TransformerModel

SOURCE_GENERATION_ANNOTATION = [
    ['source.txt', 'generation.txt', 'target.txt'],
    ['domains/jokes.source', 'jokes.generation', 'domains/jokes.target'],
    ['domains/movie_quotes.source', 'movie_quotes.generation', 'domains/movie_quotes.target'],
    ['domains/news_articles.source', 'news_articles.generation', 'domains/news_articles.target'],
    ['domains/reddit.source', 'reddit.generation', 'domains/reddit.target'],
    ['domains/twitter.source', 'twitter.generation', 'domains/twitter.target'],
    ['genders/female.source', 'female.generation', 'genders/female.target'],
    ['genders/male.source', 'male.generation', 'genders/male.target']
]

with open('gendered_test_set/source.txt', 'r') as f:
    gendered_test = f.readlines()

with open('nongendered_test_set/source.txt', 'r') as f:
    nongendered_test = f.readlines()
    

def test_model(input_folder, logs_folder, data_folder, bpe_folder, output_folder, test_set):
    gn = TransformerModel.from_pretrained(
      f'/home/sunto/fairseq/logs_checkpoints/{logs_folder}',
      checkpoint_file='checkpoint_best.pt',
      data_name_or_path=f'/home/sunto/fairseq/data-bin/{data_folder}',
      bpe='subword_nmt',
      bpe_codes=f'/home/sunto/fairseq/examples/translation/{bpe_folder}'
    )
    
    Path(f"{input_folder}/generations/{output_folder}").mkdir(parents=True, exist_ok=True)
    
    generations = gn.translate(test_set)
    with open(f'{input_folder}/generations/{output_folder}/generation.txt', 'w') as f:
        for sent in generations:
            f.write(sent)
            f.write('\n')


def create_outputs(func, output_folder):
    source_file = 'source.txt'
    generation_file = 'generation.txt'
    
    Path(f"test_set/{output_folder}").mkdir(parents=True, exist_ok=True)

    outputs = generate_outputs(source_file=os.path.join('test_set', source_file), func=func)
    with open(f"test_set/{output_folder}/{generation_file}", 'w') as f:
        for sent in outputs:
            f.write(sent)


def evaluate_outputs(output_folder, fname):
    scores = dict()
    for file_set in SOURCE_GENERATION_ANNOTATION:
        generation_file = file_set[1]
        annotation_file = file_set[2]

        results = get_metrics(generation_file=os.path.join('test_set', output_folder, generation_file),
                              annotation_file=os.path.join('test_set', annotation_file))

        domain = generation_file.split('.')[0]
        scores[domain] = results

    with open(f"test_set/scores/{fname}.json", 'w') as f:
        json.dump(scores, f)


def main():
    
    # create_outputs(func=identity, input_folder='nongendered_test_set', output_folder='generations/identity')
    #
    # from jewang_neutral_converter import jewang_convert
    # create_outputs(func=jewang_convert, input_folder='nongendered_test_set', output_folder='generations/prior_work')
    #
    # from old_smart_convert import old_convert
    # create_outputs(func=old_convert, input_folder='nongendered_test_set', output_folder='generations/old_convert_1')
    #
    # from old_score_smart_convert import convert_old_score
    # create_outputs(func=convert_old_score, input_folder='nongendered_test_set', output_folder='generations/old_convert_2')
    #
    # from smart_convert import convert
    # create_outputs(func=convert, input_folder='nongendered_test_set', output_folder='generations/convert')

    # evaluate_outputs(output_folder='generations/jewang_convert', fname='jewang_convert')
    # evaluate_outputs(output_folder='generations/old_convert_1', fname='old_convert_1')
    # evaluate_outputs(output_folder='generations/old_convert_2', fname='old_convert_2')
    # evaluate_outputs(output_folder='generations/convert', fname='convert')

#     eval_set = 'nongendered_test_set'

#     with open(f'{eval_set}/source.txt', 'r') as f:
#         source = f.readlines()
#     #
#     female_indices = [i for i, sent in enumerate(source) if is_gendered(sent) == 'female']
#     male_indices = [i for i, sent in enumerate(source) if is_gendered(sent) == 'male']
#     print(len(female_indices))
#     print(len(male_indices))

#     algorithms = ['convert', 'old_convert_2', 'old_convert_1', 'prior_work', 'identity']
#     # algorithms = ['model_5_5', 'model_6_4', 'model_7_3', 'model_8_2', 'model_9_1', 'model_10_0', 'model_full']
#     for algo in algorithms:
#         generation_fine_grained = split_generation(generation_folder=f"{algo}",
#                                                    female_indices=female_indices,
#                                                    male_indices=male_indices)

#         for domain, generation in generation_fine_grained.items():
#             with open(f'{eval_set}/generations/{algo}/{domain}.generation', 'w') as f:
#                 for sent in generation:
#                     f.write(sent)

#         evaluate_outputs(output_folder=f'generations/{algo}',
#                          fname=algo)
    
#     test_model(input_folder='nongendered',
#                logs_folder='old_files/transformer_checkpoints_0817', 
#                     data_folder='old_files/gn_wiki_data_updated_all_combined_tokenized', 
#                     bpe_folder='old_datasets/gn_wiki_data_updated_all_combined/code', 
#                     output_folder='model_full')

    test_model(input_folder='nongendered_test_set',
               logs_folder='new_datasets/simple_augmentation', 
                data_folder='new_datasets/simple_augmentation', 
                bpe_folder='new_datasets/simple_augmentation/code', 
                output_folder='model_simple_augmentation',
              test_set=nongendered_test)
    
#     for i in range(5, 11):
#         test_model(input_folder='nongendered',
#                    logs_folder=f'gn_wiki_mix/{i}_{10-i}', 
#                     data_folder=f'gn_wiki_mix/{i}_{10-i}', 
#                     bpe_folder=f'gn_wiki_mix/{i}_{10-i}/code', 
#                     output_folder=f'model_{i}_{10-i}')
    
    
#     create_outputs(func=test_model_full, output_folder='generations/model_full')
#     create_outputs(func=test_model_5_5, output_folder='generations/model_5_5')
#     create_outputs(func=test_model_6_4, output_folder='generations/model_6_4')
#     create_outputs(func=test_model_7_3, output_folder='generations/model_7_3')
#     create_outputs(func=test_model_8_2, output_folder='generations/model_8_2')
#     create_outputs(func=test_model_9_1, output_folder='generations/model_9_1')
#     create_outputs(func=test_model_10_0, output_folder='generations/model_10_0')

#     evaluate_outputs(output_folder='generations/jewang_convert', fname='jewang_convert')
#     evaluate_outputs(output_folder='generations/old_convert_1', fname='old_convert_1')
#     evaluate_outputs(output_folder='generations/old_convert_2', fname='old_convert_2')
#     evaluate_outputs(output_folder='generations/convert', fname='convert')


if __name__ == "__main__":
    main()
