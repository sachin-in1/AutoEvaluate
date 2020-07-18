from rake_nltk import Rake

r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

print(r.extract_keywords_from_text("Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered.Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generatingsets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."))
# print(r.get_ranked_phrases())
print(r.get_ranked_phrases_with_scores())

from rake_nltk import Metric, Rake


r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
r = Rake(ranking_metric=Metric.WORD_DEGREE)
r = Rake(ranking_metric=Metric.WORD_FREQUENCY)

# If you want to control the max or min words in a phrase, for it to be
# considered for ranking you can initialize a Rake instance as below:

r = Rake(min_length=2, max_length=4)

# print(r)