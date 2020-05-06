import React from "react";
import { Stack } from "office-ui-fabric-react/lib/Stack";
import { mergeStyleSets } from "@uifabric/styling";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "../Localization/localization";
import { ActionButton } from "office-ui-fabric-react/lib/Button";

export interface IIntroTabProps {
    onNext: () => void;
}

export class IntroTab extends React.PureComponent <IIntroTabProps> {
    private static readonly classNames = mergeStyleSets({
        firstSection: {
            padding: "60px 60px",
            fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
            backgroundColor: "#222222",
            color: "#FFFFFF",
        },
        firstSectionTitle: {
            maxWidth: "400px",
            fontSize: "42px",
            lineHeight: "50px",
            fontWeight: "300",
            fontStyle: "normal",
            marginBottom: "10px"
        },
        firstSectionSubtitle: {
            maxWidth: "400px",
            fontSize: "42px",
            lineHeight: "50px",
            fontWeight: "600",
            fontStyle: "normal"
        },
        firstSectionBody: {
            maxWidth: "372px",
            paddingTop: "10px",
            paddingBottom: "20px",
            color: "#EBEBEB",
            fontStyle: "normal",
            fontWeight: "normal",
            fontSize: "15px",
            lineHeight: "20px",
        },
        firstSectionGraphics: {
            width: "356px",
            height: "154px",
        },
        lowerSection: {
            padding: "50px 70px 90px 50px",
            backgroundColor: "#F2F2F2",
            color: "#333333",
            flexGrow: 1
        },
        stepsContainer: {
            display: "flex",
            flexDirection: "row",
            justifyContent: "space-between",
            borderBottom: "1px solid #CCCCCC",
            paddingBottom: "38px"
        },
        boldStep: {
            fontSize: "15px",
            lineHeight: "20px",
            fontWeight: "600",
            fontStyle: "normal",
            maxWidth: "300px",
            paddingRight: "25px",
            flex: 1,
        },
        numericLabel: {
            display:"inline-block",
            fontSize: "15px",
            lineHeight: "20px",
            fontWeight: "bold",
            fontStyle: "normal",
            color: "#5A53FF",
            width: "25px"
        },
        stepLabel: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "20px",
            fontWeight: "600",
            fontStyle: "normal"
        },
        explanatoryStep: {
            maxWidth: "300px",
            paddingRight: "20px",
            flex: 1
        },
        explanatoryText: {
            paddingTop: "1px",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            fontStyle: "normal",
            color: "#333333"
        },
        getStarted: {
            paddingTop: "0px",
            color: "#FFFFFF",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "600",
            fontStyle: "normal",
            borderRadius: "5px",
            background: "linear-gradient(348.1deg, #5A53FF -73.33%, #5A53FF 84.28%)",
            selectors: {
                '&:hover': { color: "#ffffff" }
            }
        }
    });

    render(): React.ReactNode {
        return (<Stack style={{height: "100%"}}>
            <div className={IntroTab.classNames.firstSection}>
                <Stack wrap horizontalAlign={"start"} style={{width: "100%", height: "200px"}} >
                    <div className={IntroTab.classNames.firstSectionTitle}>{localization.Intro.welcome}</div>
                    <div className={IntroTab.classNames.firstSectionSubtitle}>{localization.Intro.fairlearnDashboard}</div>
                    <div className={IntroTab.classNames.firstSectionBody}>{localization.Intro.introBody}</div>
                    <div className={IntroTab.classNames.firstSectionGraphics}>
                        <svg width="358" height="156" viewBox="0 0 358 156" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M9.5 121C14.1944 121 18 117.194 18 112.5C18 107.806 14.1944 104 9.5 104C4.80558 104 1 107.806 1 112.5C1 117.194 4.80558 121 9.5 121Z" stroke="white" stroke-width="2"/>
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M87.5 82C92.1944 82 96 78.1944 96 73.5C96 68.8056 92.1944 65 87.5 65C82.8056 65 79 68.8056 79 73.5C79 78.1944 82.8056 82 87.5 82Z" stroke="white" stroke-width="2"/>
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M169.5 155C174.194 155 178 151.194 178 146.5C178 141.806 174.194 138 169.5 138C164.806 138 161 141.806 161 146.5C161 151.194 164.806 155 169.5 155Z" stroke="white" stroke-width="2"/>
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M248.5 65C253.194 65 257 61.1944 257 56.5C257 51.8056 253.194 48 248.5 48C243.806 48 240 51.8056 240 56.5C240 61.1944 243.806 65 248.5 65Z" stroke="white" stroke-width="2"/>
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M348.5 18C353.194 18 357 14.1944 357 9.5C357 4.80558 353.194 1 348.5 1C343.806 1 340 4.80558 340 9.5C340 14.1944 343.806 18 348.5 18Z" stroke="white" stroke-width="2"/>
                            <path d="M22 108L76 81" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M98.4189 80.4179L160 136" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M178.419 135.583L239.581 66.4167" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M336.379 15.2083L260.207 49.375" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                </Stack>
            </div>
            <div className={IntroTab.classNames.lowerSection}>
                <div className={IntroTab.classNames.stepsContainer}>
                    <div className={IntroTab.classNames.boldStep}>{localization.Intro.explanatoryStep}</div>
                    <div className={IntroTab.classNames.explanatoryStep}>
                        <div>
                            <span className={IntroTab.classNames.numericLabel}>01</span>
                            <span className={IntroTab.classNames.stepLabel}>{localization.Intro.features}</span>
                        </div>
                        <div className={IntroTab.classNames.explanatoryText}>{localization.Intro.featuresInfo}</div>
                    </div>
                    <div className={IntroTab.classNames.explanatoryStep}>
                        <div>
                            <span className={IntroTab.classNames.numericLabel}>02</span>
                            <span className={IntroTab.classNames.stepLabel}>{localization.Intro.accuracy}</span>
                        </div>
                        <div className={IntroTab.classNames.explanatoryText}>{localization.Intro.accuracyInfo}</div>
                    </div>
                    <div className={IntroTab.classNames.explanatoryStep}>
                        <div>
                            <span className={IntroTab.classNames.numericLabel}>03</span>
                            <span className={IntroTab.classNames.stepLabel}>{localization.Intro.parity}</span>
                        </div>
                        <div className={IntroTab.classNames.explanatoryText}>{localization.Intro.parityInfo}</div>
                    </div>
                </div>
                <Stack horizontalAlign={"end"} style={{marginTop: "20px"}}>
                    <ActionButton 
                        iconProps={{iconName: "Forward", style: { color: 'white' }}}
                        className={IntroTab.classNames.getStarted}
                        onClick={this.props.onNext}>{localization.Intro.getStarted}</ActionButton>
                </Stack>
            </div>
        </Stack>);
    }
}