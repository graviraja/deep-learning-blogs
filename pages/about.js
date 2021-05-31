import siteMetadata from '@/data/siteMetadata'
import SocialIcon from '@/components/social-icons'
import { PageSeo } from '@/components/SEO'

export default function About() {
  return (
    <>
      <PageSeo
        title={`About - ${siteMetadata.author}`}
        description={`About me - ${siteMetadata.author}`}
        url={`${siteMetadata.siteUrl}/about`}
      />
      <div className="divide-y">
        <div className="pt-6 pb-8 space-y-2 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            About
          </h1>
        </div>
        <div className="items-start space-y-2 xl:grid xl:grid-cols-3 xl:gap-x-8 xl:space-y-0">
          <div className="flex flex-col items-center pt-8 space-x-2">
            <img src={siteMetadata.image} alt="avatar" className="w-48 h-48 rounded-full" />
            <h3 className="pt-4 pb-2 text-2xl font-bold leading-8 tracking-tight">
              {siteMetadata.author}
            </h3>
            <div className="text-gray-500 dark:text-gray-400">Founding Engineer - NLP</div>
            <div className="text-gray-500 dark:text-gray-400">Enterpret</div>
            <div className="flex pt-6 space-x-3">
              <SocialIcon kind="mail" href={`mailto:${siteMetadata.email}`} />
              <SocialIcon kind="github" href={siteMetadata.github} />
              <SocialIcon kind="linkedin" href={siteMetadata.linkedin} />
              <SocialIcon kind="twitter" href={siteMetadata.twitter} />
            </div>
          </div>
          <div className="pt-8 pb-8 prose dark:prose-dark max-w-none xl:col-span-2">
            <p>Hey there, welcome!</p>
            <p>
              I am currently working in <span href="https://www.enterpret.com/">Enterpret</span> as
              a Founding Engineer - NLP.
            </p>
            <p>
              My interests are in Unsupervised Algorithms, Semantic Similarity and Productionising
              the NLP models. I also like to follow latest research works happening in the NLP
              domain.
            </p>
            <p>
              Besides work, I like cooking{' '}
              <span role="img" aria-labelledby="cooking">
                🥘
              </span>{' '}
              , cycling{' '}
              <span role="img" aria-labelledby="cycling">
                🚴‍♀️
              </span>{' '}
              , kdramas{' '}
              <span role="img" aria-labelledby="kdramas">
                🎥
              </span>
              .
            </p>
          </div>
        </div>
      </div>
    </>
  )
}
