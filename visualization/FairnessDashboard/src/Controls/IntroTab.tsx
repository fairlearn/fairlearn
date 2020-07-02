import { ActionButton, PrimaryButton } from 'office-ui-fabric-react/lib/Button';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
import { Text } from 'office-ui-fabric-react';
import React from 'react';
import { localization } from '../Localization/localization';
import { IntroTabStyles } from './IntroTab.styles';

export interface IIntroTabProps {
    onNext: () => void;
}

export class IntroTab extends React.PureComponent<IIntroTabProps> {
    render(): React.ReactNode {
        const styles = IntroTabStyles();
        return (
            <Stack style={{ height: '100%' }}>
                <div className={styles.firstSection}>
                    <Stack wrap horizontalAlign={'start'} className={styles.firstSectionContainer}>
                        <Text className={styles.firstSectionTitle} block>
                            {localization.Intro.welcome}
                        </Text>
                        <Text className={styles.firstSectionSubtitle} block>
                            {localization.Intro.fairlearnDashboard}
                        </Text>
                        <Text className={styles.firstSectionBody} variant={'large'} block>
                            {localization.Intro.introBody}
                        </Text>
                        <div className={styles.firstSectionGraphics}>
                            <svg width="358" height="156" viewBox="0 0 358 156" xmlns="http://www.w3.org/2000/svg">
                                <path
                                    fillRule="evenodd"
                                    clipRule="evenodd"
                                    d="M9.5 121C14.1944 121 18 117.194 18 112.5C18 107.806 14.1944 104 9.5 104C4.80558 104 1 107.806 1 112.5C1 117.194 4.80558 121 9.5 121Z"
                                    stroke="white"
                                    strokeWidth="2"
                                />
                                <path
                                    fillRule="evenodd"
                                    clipRule="evenodd"
                                    d="M87.5 82C92.1944 82 96 78.1944 96 73.5C96 68.8056 92.1944 65 87.5 65C82.8056 65 79 68.8056 79 73.5C79 78.1944 82.8056 82 87.5 82Z"
                                    stroke="white"
                                    strokeWidth="2"
                                />
                                <path
                                    fillRule="evenodd"
                                    clipRule="evenodd"
                                    d="M169.5 155C174.194 155 178 151.194 178 146.5C178 141.806 174.194 138 169.5 138C164.806 138 161 141.806 161 146.5C161 151.194 164.806 155 169.5 155Z"
                                    stroke="white"
                                    strokeWidth="2"
                                />
                                <path
                                    fillRule="evenodd"
                                    clipRule="evenodd"
                                    d="M248.5 65C253.194 65 257 61.1944 257 56.5C257 51.8056 253.194 48 248.5 48C243.806 48 240 51.8056 240 56.5C240 61.1944 243.806 65 248.5 65Z"
                                    stroke="white"
                                    strokeWidth="2"
                                />
                                <path
                                    fillRule="evenodd"
                                    clipRule="evenodd"
                                    d="M348.5 18C353.194 18 357 14.1944 357 9.5C357 4.80558 353.194 1 348.5 1C343.806 1 340 4.80558 340 9.5C340 14.1944 343.806 18 348.5 18Z"
                                    stroke="white"
                                    strokeWidth="2"
                                />
                                <path d="M22 108L76 81" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                <path
                                    d="M98.4189 80.4179L160 136"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                                <path
                                    d="M178.419 135.583L239.581 66.4167"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                                <path
                                    d="M336.379 15.2083L260.207 49.375"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                            </svg>
                        </div>
                    </Stack>
                </div>
                <div className={styles.lowerSection}>
                    <div className={styles.stepsContainer}>
                        <Text variant={'large'} className={styles.boldStep}>
                            {localization.Intro.explanatoryStep}
                        </Text>
                        <div className={styles.explanatoryStep}>
                            <div>
                                <Text variant={'large'} className={styles.numericLabel}>
                                    01
                                </Text>
                                <Text variant={'large'}>{localization.Intro.features}</Text>
                            </div>
                            <Text className={styles.explanatoryText} block>
                                {localization.Intro.featuresInfo}
                            </Text>
                        </div>
                        <div className={styles.explanatoryStep}>
                            <div>
                                <Text variant={'large'} className={styles.numericLabel}>
                                    02
                                </Text>
                                <Text variant={'large'}>{localization.Intro.accuracy}</Text>
                            </div>
                            <Text className={styles.explanatoryText} block>
                                {localization.Intro.accuracyInfo}
                            </Text>
                        </div>
                        <div className={styles.explanatoryStep}>
                            <div>
                                <Text variant={'large'} className={styles.numericLabel}>
                                    03
                                </Text>
                                <Text variant={'large'}>{localization.Intro.parity}</Text>
                            </div>
                            <Text className={styles.explanatoryText} block>
                                {localization.Intro.parityInfo}
                            </Text>
                        </div>
                    </div>
                    <Stack horizontalAlign={'end'} style={{ marginTop: '20px' }}>
                        <PrimaryButton className={styles.getStarted} onClick={this.props.onNext}>
                            {localization.Intro.getStarted}
                        </PrimaryButton>
                    </Stack>
                </div>
            </Stack>
        );
    }
}
