import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { Stack } from "office-ui-fabric-react/lib/Stack";
import React from "react";
import { localization } from "../Localization/localization";
import { IntroTabStyles } from "./IntroTab.styles";

export interface IIntroTabProps {
    onNext: () => void;
}

export class IntroTab extends React.PureComponent <IIntroTabProps> {
    // private static readonly classNames = mergeStyleSets({
    //     firstSection: {
    //         padding: "43px 94px",
    //         fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
    //         backgroundColor: "#333333",
    //         color: "#FFFFFF",
    //     },
    //     firstSectionTitle: {
    //         fontSize: "60px",
    //         lineHeight: "82px",
    //         fontWeight: "100"
    //     },
    //     firstSectionSubtitle: {
    //         fontSize: "60px",
    //         lineHeight: "82px",
    //         fontWeight: "500"
    //     },
    //     firstSectionBody: {
    //         paddingTop: "30px",
    //         paddingBottom: "70px",
    //         maxWidth: "500px",
    //         color: "#EBEBEB",
    //         fontWeight: "300",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //     },
    //     lowerSection: {
    //         padding: "50px 70px 90px 90px",
    //         backgroundColor: "#F2F2F2",
    //         color: "#333333",
    //         flexGrow: 1
    //     },
    //     stepsContainer: {
    //         display: "flex",
    //         flexDirection: "row",
    //         justifyContent: "space-between",
    //         borderBottom: "1px solid #CCCCCC",
    //         paddingBottom: "38px"
    //     },
    //     boldStep: {
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "500",
    //         maxWidth: "300px",
    //         paddingRight: "25px",
    //         flex: 1,
    //     },
    //     numericLabel: {
    //         display:"inline-block",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "700",
    //         color: "#000000",
    //         width: "30px"
    //     },
    //     stepLabel: {
    //         color: "#333333",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "500",
    //     },
    //     explanatoryStep: {
    //         maxWidth: "300px",
    //         paddingRight: "20px",
    //         flex: 1
    //     },
    //     explanatoryText: {
    //         paddingTop: "15px",
    //         fontSize: "15px",
    //         lineHeight: "20px",
    //         color: "#666666"
    //     },
    //     getStarted: {
    //         paddingTop: "30px",
    //         color: "#333333",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "500"
    //     }
    // });

    render(): React.ReactNode {
        const styles = IntroTabStyles();
        return (<Stack style={{height: "100%"}}>
            <div className={styles.firstSection}>
                <div className={styles.firstSectionTitle}>{localization.Intro.welcome}</div>
                <div className={styles.firstSectionSubtitle}>{localization.Intro.fairlearnDashboard}</div>
                <div className={styles.firstSectionBody}>{localization.Intro.introBody}</div>
            </div>
            <div className={styles.lowerSection}>
                <div className={styles.stepsContainer}>
                    <div className={styles.boldStep}>{localization.Intro.explanatoryStep}</div>
                    <div className={styles.explanatoryStep}>
                        <div>
                            <span className={styles.numericLabel}>01</span>
                            <span className={styles.stepLabel}>{localization.Intro.features}</span>
                        </div>
                        <div className={styles.explanatoryText}>{localization.Intro.featuresInfo}</div>
                    </div>
                    <div className={styles.explanatoryStep}>
                        <div>
                            <span className={styles.numericLabel}>02</span>
                            <span className={styles.stepLabel}>{localization.Intro.accuracy}</span>
                        </div>
                        <div className={styles.explanatoryText}>{localization.Intro.accuracyInfo}</div>
                    </div>
                </div>
                <Stack horizontalAlign={"center"}>
                    <ActionButton 
                        iconProps={{iconName: "Forward"}}
                        className={styles.getStarted}
                        onClick={this.props.onNext}>{localization.Intro.getStarted}</ActionButton>
                </Stack>
            </div>
        </Stack>);
    }
}