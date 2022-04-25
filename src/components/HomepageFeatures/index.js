import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'System Design',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Hand written system design notes and popular system design problem solutions from open-source platforms.
      </>
    ),
  },
  {
    title: 'Interview',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Data Science and Machine Learning Interview Questions with their Solutions.
      </>
    ),
  },
  {
    title: 'MLOps',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        MLOps notes from all free resources and many hand written notes on some important MLOps concepts.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
