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
            padding: "30px 50px",
            fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
            backgroundColor: "#333",
            color: "#FFF",
        },
        firstSectionTitle: {
            fontSize: "40px",
        },
        firstSectionSubtitle: {
            fontSize: "43px"
        },
        firstSectionBody: {
            maxWidth: "500px"
        },
        lowerSection: {
            padding: "20px 50px",
            backgroundColor: "#EEE",
            color: "#222",
            flexGrow: 1
        },
        stepsContainer: {
            borderBottom: "1px solid grey",
            paddingBottom: "20px"
        },
        explanatoryStep: {
            fontSize: "18px",
            minWidth: "300px"
        },
        getStarted: {
            paddingTop: "30px",
            fontSize: "30px"
        }
    });

    render(): React.ReactNode {
        return (<Stack style={{height: "100%"}}>
            <div className={IntroTab.classNames.firstSection}>
                <Text block className={IntroTab.classNames.firstSectionTitle}>{localization.Intro.welcome}</Text>
                <Text block className={IntroTab.classNames.firstSectionSubtitle}>{localization.Header.title}</Text>
                <Text block className={IntroTab.classNames.firstSectionBody}>{localization.loremIpsum}</Text>
            </div>
            <div className={IntroTab.classNames.lowerSection}>
                <Stack horizontal horizontalAlign={"space-between"} className={IntroTab.classNames.stepsContainer}>
                    <Text className={IntroTab.classNames.explanatoryStep}>{localization.Intro.explanatoryStep}</Text>
                    <Text className={IntroTab.classNames.explanatoryStep}>{localization.Intro.explanatoryStep}</Text>
                    <Text className={IntroTab.classNames.explanatoryStep}>{localization.Intro.explanatoryStep}</Text>
                </Stack>
                <Stack horizontalAlign={"center"}>
                    <ActionButton 
                        iconProps={{iconName: "Forward"}}
                        className={IntroTab.classNames.getStarted}
                        onClick={this.props.onNext}>{localization.Intro.getStarted}</ActionButton>
                </Stack>
            </div>
        </Stack>);
    }
}