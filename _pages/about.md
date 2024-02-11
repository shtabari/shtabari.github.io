---
permalink: /
title: "About"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---


<!-- ![Persian Fibonacci](/images/homepage_pic1.png){: .align-right width="300px"} -->
<img src="/images/homepage_pic1.png" width="350" style="float:right">
Greetings! My name is Hamid, and I currently hold the position of Senior Data Scientist within Walmart's Global Tech division. Following the completion of my Master's degree in Statistics from the University of Arkansas' Mathematics department, I embarked on my professional journey with Walmart in the summer of 2020, initially as an intern.  This opportunity quickly transitioned into a full-time role immediately following my internship. My work involves a variety of projects, including the development of recommendation systems through the use of computer vision, the creation of a scalable forecasting engine, demand forecasting utilizing deep learning models, and projects involving large language models and natural language processing (NLP). 


Professional Experience in the Industry
======

In my role, I have spearheaded various impactful projects leveraging advanced technologies and methodologies. My expertise in Language Model Optimization, particularly with LLMs and NLP, was pivotal in a project for Walmart where I specialized in converting textual information into SQL code. This initiative significantly accelerated the provision of valuable insights to merchants by optimizing Language Model Models on Walmart datasets. 

In another major endeavor, I led the Financial-Planning project for the Walmart Team, utilizing BigQuery, Cloud Computing, Tableau, Kubernetes, and Machine Learning Engineering skills. My leadership in this project produced ML-driven accurate quarterly forecasts for Omni Walmart departments, achieving remarkable outcomes including a quarterly saving of 4 million dollars and 152 hours per associate.

Furthermore, my proficiency in TensorFlow and Spark was instrumental in the development of a deep learning sequence model aimed at predicting item-level forecasts at each store. This project was monumental, generating 540 million forecast units, which translates to a staggering 9 billion dollars annually. Additionally, I designed and implemented a forecasting framework/pipeline that automated geo-demand forecasts, showcasing my comprehensive skills in managing and executing complex projects that drive significant financial and operational efficiencies.


Overview of My Academic Journey
======

From a young age, I've had a profound interest in mathematics, consistently achieving top marks in math and physics throughout my high school career. Working as a computer technician during my summers fueled my fascination with technology, leading me to self-study computer science fundamentals. In university, I was entrusted with managing a GPU server, an important role that not only enabled me to support my community, including researchers and graduate students, but also to author a significant paper on drug discovery using quantum mechanics.

My journey brought me to the United States to pursue a PhD in computational biophysics, a field I found incredibly intriguing for its use of computer simulations to decode biological phenomena. Nonetheless, my enduring love for mathematics and my exceptional performance in graduate-level statistics courses steered me towards a scholarly path in statistics.

<!-- A data-driven personal website
======
Like many other Jekyll-based GitHub Pages templates, academicpages makes you separate the website's content from its form. The content & metadata of your website are in structured markdown files, while various other files constitute the theme, specifying how to transform that content & metadata into HTML pages. You keep these various markdown (.md), YAML (.yml), HTML, and CSS files in a public GitHub repository. Each time you commit and push an update to the repository, the [GitHub pages](https://pages.github.com/) service creates static HTML pages based on these files, which are hosted on GitHub's servers free of charge.

Many of the features of dynamic content management systems (like Wordpress) can be achieved in this fashion, using a fraction of the computational resources and with far less vulnerability to hacking and DDoSing. You can also modify the theme to your heart's content without touching the content of your site. If you get to a point where you've broken something in Jekyll/HTML/CSS beyond repair, your markdown files describing your talks, publications, etc. are safe. You can rollback the changes or even delete the repository and start over -- just be sure to save the markdown files! Finally, you can also write scripts that process the structured data on the site, such as [this one](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb) that analyzes metadata in pages about talks to display [a map of every location you've given a talk](https://academicpages.github.io/talkmap.html).

Getting started
======
1. Register a GitHub account if you don't have one and confirm your e-mail (required!)
1. Fork [this repository](https://github.com/academicpages/academicpages.github.io) by clicking the "fork" button in the top right. 
1. Go to the repository's settings (rightmost item in the tabs that start with "Code", should be below "Unwatch"). Rename the repository "[your GitHub username].github.io", which will also be your website's URL.
1. Set site-wide configuration and create content & metadata (see below -- also see [this set of diffs](http://archive.is/3TPas) showing what files were changed to set up [an example site](https://getorg-testacct.github.io) for a user with the username "getorg-testacct")
1. Upload any files (like PDFs, .zip files, etc.) to the files/ directory. They will appear at https://[your GitHub username].github.io/files/example.pdf.  
1. Check status by going to the repository settings, in the "GitHub pages" section

Site-wide configuration
------
The main configuration file for the site is in the base directory in [_config.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_config.yml), which defines the content in the sidebars and other site-wide features. You will need to replace the default variables with ones about yourself and your site's github repository. The configuration file for the top menu is in [_data/navigation.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_data/navigation.yml). For example, if you don't have a portfolio or blog posts, you can remove those items from that navigation.yml file to remove them from the header. 

Create content & metadata
------
For site content, there is one markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For example, each talk is a markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory).

**Markdown generator**

I have also created [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual markdown files that will be properly formatted for the academicpages template. The sample CSVs in that directory are the ones I used to create my own personal website at stuartgeiger.com. My usual workflow is that I keep a spreadsheet of my publications and talks, then run the code in these notebooks to generate the markdown files, then commit and push them to the GitHub repository.

How to edit your site's GitHub repository
------
Many people use a git client to create files on their local computer and then push them to GitHub's servers. If you are not familiar with git, you can directly edit these configuration and markdown files directly in the github.com interface. Navigate to a file (like [this one](https://github.com/academicpages/academicpages.github.io/blob/master/_talks/2012-03-01-talk-1.md) and click the pencil icon in the top right of the content preview (to the right of the "Raw | Blame | History" buttons). You can delete a file by clicking the trashcan icon to the right of the pencil icon. You can also create new files or upload files by navigating to a directory and clicking the "Create new file" or "Upload files" buttons. 

Example: editing a markdown file for a talk
![Editing a markdown file for a talk](/images/editing-talk.png)

For more info
------
More info about configuring academicpages can be found in [the guide](https://academicpages.github.io/markdown/). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful. -->
